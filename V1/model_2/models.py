


from dataclasses import dataclass


import torch
import torch.nn as nn

from transformers import HubertModel, HubertConfig


from helper_fns import _init_weights, align_length_with_directly_insert
from losses import WingLoss, L1Loss, WingFlattenLoss, L1FlattenLoss
from data_prepare import resample_feature_from_50_to_60


@dataclass
class Model2Config:

    pretrained_model_or_path: str = "TencentGameMate/chinese-hubert-base"

    n_lumi_mouth_channels: int = 52
    n_lumi_eye_channels: int = 14




# TODO 添加特殊的 attention mask；

class Model2(nn.Module):

    def __init__(self, model_config: Model2Config, training_config: "TrainingConfig") -> None:
        super().__init__()

        self.pretrained_model_config = HubertConfig.from_pretrained(model_config.pretrained_model_or_path)
        self.pretrained_model = HubertModel.from_pretrained(model_config.pretrained_model_or_path)
        self.hidden_size = self.pretrained_model_config.hidden_size

        self.mouth_head = nn.Linear(in_features=self.hidden_size, out_features=model_config.n_lumi_mouth_channels)
        self.eye_head = nn.Linear(in_features=self.hidden_size, out_features=model_config.n_lumi_eye_channels)
        self.mouth_head.apply(_init_weights)
        self.eye_head.apply(_init_weights)

        self.loss_config = training_config.loss_config
        self.wing_loss_config = self.loss_config.wing_loss_config
        self.wing_loss_fn = WingFlattenLoss(omega=self.wing_loss_config.omega, epsilon=self.wing_loss_config.epsilon, emoji_weight=self.wing_loss_config.emoji_weight)

        self.l1_loss_fn = L1FlattenLoss()

        self.model_config = model_config
        self.training_config = training_config


    def forward(self, batch):
        net_input = batch["net_input"]
        model_outputs = self.pretrained_model(
            input_values=net_input["audios"],
            attention_mask=net_input["padding_mask"],
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

        real_hidden_states = []
        real_ctrl_labels = []
        for i in range(len(audio_feature_lengths)):
            cur_feature_length = audio_feature_lengths[i]
            cur_hidden_state = encoder_hidden_state[i]
            cur_hidden_state = cur_hidden_state[:cur_feature_length]

            label_length = ctrl_label_lengths[i]
            if cur_feature_length > label_length:
                cur_hidden_state = cur_hidden_state[:label_length]
            elif cur_feature_length < label_length:
                cur_hidden_state = align_length_with_directly_insert(cur_hidden_state, label_length)
            
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
                mouth_wing_loss, mouth_wing_record_loss, mouth_wing_record_num = self.wing_loss_fn(mouth_ctrl_pred, mouth_ctrl_labels)
                loss += self.training_config.loss_config.wing_loss_weight * mouth_wing_loss
            
            if self.training_config.loss_config.use_l1_loss:
                mouth_l1_loss, mouth_l1_record_loss, mouth_l1_record_num  = self.l1_loss_fn(mouth_ctrl_pred, mouth_ctrl_labels)
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





