


from dataclasses import dataclass
from typing import Optional, List


import torch
import torch.nn as nn

from .modified_hubert import HubertModel, HubertConfig

from transformers import HubertModel as TransHubertModel
from transformers import HubertConfig as TransHubertConfig

from helper_fns import _init_weights, align_length_with_directly_insert
from losses import WingLoss, L1Loss, WingFlattenLoss, L1FlattenLoss
from data_prepare import resample_feature_from_50_to_60


@dataclass
class Model2Config:

    pretrained_model_or_path: str = "TencentGameMate/chinese-hubert-base"

    n_lumi_mouth_channels: int = 52
    n_lumi_eye_channels: int = 14


@dataclass
class SpecializedPretrainedModelConfig:
    """
    1. 为了方便迭代，目前所有额外的特征都直接添加到 TrainingConfig 中，即使这些特征的参数应该由 ModelConfig 或者 DatasetConfig 进行设置；
    2. 为了定制化 hubert，例如在中间层添加额外的 feature embedding，我们需要传入额外的参数配置，为了能够更直观地对模型进行修改和定制，这里单独写一个 config；
    3. 该 config 的具体的参数的值全部从 TrainingConfig 中获取，其所有默认值均为 None；
    
    """

    add_phn_embedding: Optional[bool] = None
    phn_num: Optional[int] = None
    phn_padding_idx: Optional[int] = None
    phn_layer: Optional[int] = None  # 将该 embedding 加在模型的哪一个具体的层上，例如第 1 层就是和时序卷积的输出相加送进 transfomrer，12 层就是添加到第 11 层的输出上；

    add_emotion_embedding: Optional[bool] = None
    emotion_num: Optional[int] = None
    emotion_layer: Optional[int] = None

    # 特殊的 mask；
    add_double_casual_mask: Optional[bool] = None
    dcm_ratios: Optional[List] = None  # List[float]




# TODO 添加特殊的 attention mask；

class Model2(nn.Module):

    def __init__(self, model_config: Model2Config, training_config: "TrainingConfig") -> None:
        super().__init__()

        phn_config = training_config.phn_config
        emotion_config = training_config.emotion_config
        casual_mask_config = training_config.casual_mask_config
        specialized_pretrained_model_config = SpecializedPretrainedModelConfig(
            add_phn_embedding=phn_config.add_phn_embedding,
            phn_num=phn_config.phn_num,
            phn_padding_idx=phn_config.phn_padding_idx,
            phn_layer=phn_config.phn_layer,

            add_emotion_embedding=emotion_config.add_emotion_embedding,
            emotion_num=emotion_config.emotion_num,
            emotion_layer=emotion_config.emotion_layer,

            add_double_casual_mask=casual_mask_config.add_double_casual_mask,
            dcm_ratios=casual_mask_config.dcm_ratios     
        )

        self.pretrained_model_config = HubertConfig.from_pretrained(model_config.pretrained_model_or_path)
        pretrained_model = TransHubertModel.from_pretrained(model_config.pretrained_model_or_path)
        self.pretrained_model = HubertModel(self.pretrained_model_config, specialized_config=specialized_pretrained_model_config)
        self.pretrained_model.load_state_dict(pretrained_model.state_dict(), strict=False)

        self.hidden_size = self.pretrained_model_config.hidden_size

        if casual_mask_config.add_double_casual_mask:
            assert len(casual_mask_config.dcm_ratios) == self.pretrained_model_config.num_hidden_layers

        # 加入用于表示扩增重复的 embedding，因为我们现在的重复扩增的插入方式大概率不会导致对于一个 hidden state 的连续的重复，因此这里只需要一个 embedding 即可；
        self.register_parameter("repeated_position_embedding", nn.Parameter(torch.normal(0, 0.02, (self.hidden_size, ))))
        # self.repeated_position_embedding = None

        self.mouth_head = nn.Linear(in_features=self.hidden_size, out_features=model_config.n_lumi_mouth_channels)
        self.eye_head = nn.Linear(in_features=self.hidden_size, out_features=model_config.n_lumi_eye_channels)
        # 一定要注意不能在 pretrained model 后直接对所有模块参数初始化；
        self.mouth_head.apply(_init_weights)
        self.eye_head.apply(_init_weights)

        self.loss_config = training_config.loss_config
        self.wing_loss_config = self.loss_config.wing_loss_config

        self.wing_flatten_loss_fn = WingFlattenLoss(omega=self.wing_loss_config.omega, epsilon=self.wing_loss_config.epsilon, emoji_weight=self.wing_loss_config.emoji_weight)
        self.l1_flatten_loss_fn = L1FlattenLoss()

        self.wing_loss_fn = WingLoss(omega=self.wing_loss_config.omega, epsilon=self.wing_loss_config.epsilon, emoji_weight=self.wing_loss_config.emoji_weight)
        self.l1_loss_fn = L1Loss()

        self.model_config = model_config
        self.training_config = training_config
        self.specialized_pretrained_model_config = specialized_pretrained_model_config


    def forward(self, batch):
        net_input = batch["net_input"]
        emotion_ids = batch.get("emotion_ids", None)
        phn_dict = batch.get("phn_dict", None)
        model_outputs = self.pretrained_model(
            input_values=net_input["audios"],
            attention_mask=net_input["padding_mask"],
            
            label_lengths=batch["label_lengths"],  # if self.specialized_pretrained_model_config.add_phn_embedding else None,
            audio_feature_lengths=net_input["audio_feature_lengths"],
            repeated_position_embedding=self.repeated_position_embedding,
            phn_dict=phn_dict,
            emotion_ids=emotion_ids,
        )
        return model_outputs

    def train_step(self, batch):
        """ 
        RawAudioCollater

        collated_batch = {
            "idx": torch.LongTensor([s['idx'] for s in batch]),
            "net_input": {
                "audios": collated_audios,
                "padding_mask": padding_mask,
                "audio_feature_lengths": audio_feature_lengths
            },
            "label_lengths": lengths,
            "label_ntokens": ntokens,
            "ctrl_labels": collated_ctrl_labels
        }
        
        注意需要对 hubert 的输出：1. 先从 50 hz 采样到 60；2. 和 ctrl label 的长度对齐；
        
        """

        model_outputs = self(batch)
        encoder_hidden_state = model_outputs.last_hidden_state
        audio_feature_lengths = batch["net_input"]["audio_feature_lengths"]
        ctrl_labels = batch["ctrl_labels"]
        ctrl_label_lengths = batch["label_lengths"]

        if not self.specialized_pretrained_model_config.add_phn_embedding:
            real_hidden_states = []
            real_ctrl_labels = []
            for i in range(len(audio_feature_lengths)):
                cur_feature_length = audio_feature_lengths[i]
                cur_hidden_state = encoder_hidden_state[i]
                cur_hidden_state = cur_hidden_state[:cur_feature_length]

                label_length = int(ctrl_label_lengths[i].item())
                if cur_feature_length > label_length:
                    cur_hidden_state = cur_hidden_state[:label_length]
                elif cur_feature_length < label_length:
                    cur_hidden_state = align_length_with_directly_insert(cur_hidden_state, label_length, self.repeated_position_embedding)
                
                real_hidden_states.append(cur_hidden_state)
                real_ctrl_labels.append(ctrl_labels[i][:label_length])
            
            real_hidden_states = torch.cat(real_hidden_states)
            real_ctrl_labels = torch.cat(real_ctrl_labels)

            loss = 0.
            # mouth loss
            mouth_wing_loss = None
            mouth_l1_loss = None
            if self.training_config.learn_mouth:
                muoth_decoder_hidden_states = real_hidden_states
                mouth_ctrl_pred = self.mouth_head(muoth_decoder_hidden_states)
                mouth_ctrl_labels = real_ctrl_labels[..., :self.model_config.n_lumi_mouth_channels]

                if self.training_config.loss_config.use_wing_loss:
                    mouth_wing_loss, mouth_wing_record_loss, mouth_wing_record_num = self.wing_flatten_loss_fn(mouth_ctrl_pred, mouth_ctrl_labels)
                    loss += self.training_config.loss_config.wing_loss_weight * mouth_wing_loss
                
                if self.training_config.loss_config.use_l1_loss:
                    mouth_l1_loss, mouth_l1_record_loss, mouth_l1_record_num  = self.l1_flatten_loss_fn(mouth_ctrl_pred, mouth_ctrl_labels)
                    loss += self.training_config.loss_config.l1_loss_weight * mouth_l1_loss
        else:
            loss = 0.
            # mouth loss
            mouth_wing_loss = None
            mouth_l1_loss = None
            # change attention mask
            attention_mask = torch.zeros(
                (len(encoder_hidden_state), encoder_hidden_state.size(1)), dtype=torch.bool, device=encoder_hidden_state.device
            )
            # these two operations makes sure that all values before the output lengths idxs are attended to
            attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), ctrl_label_lengths - 1)] = 1
            attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

            if self.training_config.learn_mouth:
                muoth_decoder_hidden_states = encoder_hidden_state
                mouth_ctrl_pred = self.mouth_head(muoth_decoder_hidden_states)
                mouth_ctrl_labels = ctrl_labels[..., :self.model_config.n_lumi_mouth_channels]

                if self.training_config.loss_config.use_wing_loss:
                    mouth_wing_loss, mouth_wing_record_loss, mouth_wing_record_num = self.wing_loss_fn(mouth_ctrl_pred, mouth_ctrl_labels, attention_mask)
                    loss += self.training_config.loss_config.wing_loss_weight * mouth_wing_loss
                
                if self.training_config.loss_config.use_l1_loss:
                    mouth_l1_loss, mouth_l1_record_loss, mouth_l1_record_num  = self.l1_loss_fn(mouth_ctrl_pred, mouth_ctrl_labels, attention_mask)
                    loss += self.training_config.loss_config.l1_loss_weight * mouth_l1_loss

        

        return {
            "loss": loss,
            "mouth_wing_loss_record": (mouth_wing_record_loss, mouth_wing_record_num) if mouth_wing_loss else None,
            "mouth_l1_loss_record": (mouth_l1_record_loss, mouth_l1_record_num) if mouth_l1_loss else None,
        }


    def inference_step(self, batch):
        
        model_outputs = self(batch)

        audio = batch["net_input"]["audios"]
        predicted_label_length = round(audio.size(1) / 16000 * 60)
        encoder_hidden_state = model_outputs.last_hidden_state[0]
        encoder_hidden_state = align_length_with_directly_insert(encoder_hidden_state, predicted_label_length)

        mouth_ctrl_pred = None
        if self.training_config.learn_mouth:
            mouth_ctrl_pred = self.mouth_head(encoder_hidden_state)
        eye_ctrl_pred = None
        if self.training_config.learn_eye:
            eye_ctrl_pred = self.eye_head(encoder_hidden_state)

        return {
            "mouth_ctrl_pred": mouth_ctrl_pred.unsqueeze(0) if mouth_ctrl_pred is not None else None,
            "eye_ctrl_pred": eye_ctrl_pred.unsqueeze(0) if eye_ctrl_pred is not None else None
        }





