

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from .modules import EEGTransformer, PhnModel
from losses import WingLoss, L1Loss

from helper_fns import _init_weights




@dataclass
class Model1Config:
    """
    static feature -> encoder (transformer encoder) -> decoder (transformer decoder);
    """

    n_lumi_mouth_channels: int = 52
    n_lumi_eye_channels: int = 14
    hidden_size: int = 384  # hidden size
    # max_seq_len: 2048
    static_feature_dim: int = 1024
    pre_lnorm: bool = True

    encoder_layer_num: int = 6
    encoder_head_num: int = 1
    encoder_head_dim: int = 64
    encoder_conv1d_filter_size: int = 1536
    encoder_conv1d_kernel_size: int = 3
    # encoder_output_size: 384
    encoder_dropout_p: float = 0.1
    encoder_dropatt_p: float = 0.1
    encoder_dropemb_p: float = 0.0

    # mouth
    mouth_decoder_layer_num: int = 5
    mouth_decoder_head_num: int = 1
    mouth_decoder_head_dim: int = 64
    mouth_decoder_conv1d_filter_size: int = 1536
    mouth_decoder_conv1d_kernel_size: int = 3
    mouth_decoder_dropout_p: float = 0.1
    mouth_decoder_dropatt_p: float = 0.1
    mouth_decoder_dropemb_p: float = 0.0

    # eye
    eye_decoder_layer_num: int = 2
    eye_decoder_head_num: int = 2
    eye_decoder_head_dim: int = 64
    eye_decoder_conv1d_filter_size: int = 1536
    eye_decoder_conv1d_kernel_size: int = 3
    eye_decoder_dropout_p: float = 0.1
    eye_decoder_dropatt_p: float = 0.1
    eye_decoder_dropemb_p: float = 0.0

    # phn config 在 training config 中进行设置；

    # pca config 在 training config 中进行设置；
    



@dataclass
class Model1Output:
    encoder_hidden_states: Any = None
    mouth_decoder_hidden_states: Any = None
    eye_decoder_hidden_states: Any = None



class Model1(nn.Module):
    """
    对应 Model1Config；
    直接实现成 module 方便处理，包括参数保存等；
    
    """

    def __init__(self, config: Model1Config, training_config: "TrainingConfig") -> None:
        super().__init__()

        self.pre_proj = nn.Linear(in_features=config.static_feature_dim, out_features=config.hidden_size)

        self.encoder = EEGTransformer(
            n_layer=config.encoder_layer_num, 
            n_head=config.encoder_head_num, 
            d_model=config.hidden_size, 
            d_head=config.encoder_head_dim, 
            d_inner=config.encoder_conv1d_filter_size, 
            kernel_size=config.encoder_conv1d_kernel_size,
            dropout=config.encoder_dropout_p, 
            dropatt=config.encoder_dropatt_p, 
            dropemb=config.encoder_dropemb_p, 
            pre_lnorm=config.pre_lnorm
        )

        assert training_config.learn_mouth or training_config.learn_eye

        if training_config.learn_mouth:
            self.mouth_decoder = EEGTransformer(
                n_layer=config.mouth_decoder_layer_num, 
                n_head=config.mouth_decoder_head_num, 
                d_model=config.hidden_size, 
                d_head=config.mouth_decoder_head_dim, 
                d_inner=config.mouth_decoder_conv1d_filter_size, 
                kernel_size=config.mouth_decoder_conv1d_kernel_size,
                dropout=config.mouth_decoder_dropout_p, 
                dropatt=config.mouth_decoder_dropatt_p, 
                dropemb=config.mouth_decoder_dropemb_p, 
                pre_lnorm=config.pre_lnorm
            )
            self.mouth_head = nn.Linear(in_features=config.hidden_size, out_features=config.n_lumi_mouth_channels)

        if training_config.learn_eye:
            self.eye_decoder = EEGTransformer(
                n_layer=config.eye_decoder_layer_num, 
                n_head=config.eye_decoder_head_num, 
                d_model=config.hidden_size, 
                d_head=config.eye_decoder_head_dim, 
                d_inner=config.eye_decoder_conv1d_filter_size, 
                kernel_size=config.eye_decoder_conv1d_kernel_size,
                dropout=config.eye_decoder_dropout_p, 
                dropatt=config.eye_decoder_dropatt_p, 
                dropemb=config.eye_decoder_dropemb_p, 
                pre_lnorm=config.pre_lnorm
            )
            self.eye_head = nn.Linear(in_features=config.hidden_size, out_features=config.n_lumi_eye_channels)
        
        # phn embedding
        phn_embedding_config=training_config.phn_embedding_config
        if phn_embedding_config.add_phn:
            self.phn_model = PhnModel(
                n_layer=phn_embedding_config.phn_layer_num, 
                n_head=phn_embedding_config.phn_head_num, 
                d_model=config.hidden_size, 
                d_head=phn_embedding_config.phn_head_dim, 
                d_inner=phn_embedding_config.phn_conv1d_filter_size, 
                kernel_size=phn_embedding_config.phn_conv1d_kernel_size,
                dropout=phn_embedding_config.phn_dropout_p, 
                dropatt=phn_embedding_config.phn_dropatt_p, 
                dropemb=phn_embedding_config.phn_dropemb_p,
                phn_num=phn_embedding_config.phn_num,
                hidden_size=config.hidden_size,
                padding_idx=phn_embedding_config.phn_padding_idx
            )

        pca_config = training_config.pca_config
        if pca_config.learn_pca:
            self.pca_head = nn.Sequential(
                nn.Linear(in_features=config.n_lumi_mouth_channels + config.n_lumi_eye_channels, out_features=pca_config.n_pca_channels),
                nn.ReLU(),
                nn.Linear(in_features=pca_config.n_pca_channels, out_features=pca_config.n_pca_channels)
            )
        
        emotion_config = training_config.emotion_config
        if emotion_config.add_emotion_embedding:
            self.emotion_embedding = nn.Embedding(
                num_embeddings=emotion_config.emotion_num,
                embedding_dim=config.hidden_size,
                padding_idx=0
            )

         # 注意如果之后开始使用预训练模型，那么需要注意随机初始化的位置；
        self.apply(_init_weights)

        self.loss_config = training_config.loss_config
        self.wing_loss_config = self.loss_config.wing_loss_config
        self.wing_loss_fn = WingLoss(omega=self.wing_loss_config.omega, epsilon=self.wing_loss_config.epsilon, emoji_weight=self.wing_loss_config.emoji_weight)

        self.l1_loss_fn = L1Loss()

        self.config = config
        self.training_config = training_config


    def forward(self, batch):
        net_input = batch["net_input"]
        audio_features = net_input["audio_features"]
        padding_mask = net_input["padding_mask"]

        # encoder
        audio_features = self.pre_proj(audio_features)
        encoder_hidden_states, _ = self.encoder(audio_features, padding_mask=padding_mask)

        # mouth decoder
        mouth_decoder_hidden_states = None
        if self.training_config.learn_mouth:
            mouth_encoder_hidden_states = encoder_hidden_states
            if self.training_config.phn_embedding_config.add_phn:
                phn_dict = batch["phn_dict"]
                phn_hidden_states = self.phn_model(
                    phns=phn_dict["collated_phns"],
                    phn_list=phn_dict["collated_phn_lists"], 
                    frame_length_list=phn_dict["collated_frame_length_lists"], 
                    padding_mask=phn_dict["phn_padding_mask"],
                )
                mouth_encoder_hidden_states = mouth_encoder_hidden_states + phn_hidden_states
            
            if self.training_config.emotion_config.add_emotion_embedding:
                # TODO 这里的输入形式之后可能会大改，为了加入 emotion prediction task；
                emotion_indices = batch["emotion_indices"]  # [B]
                emotion_embeddings = self.emotion_embedding(emotion_indices)  # [B, H]
                emotion_embeddings = emotion_embeddings.unsqueeze(1).repeat(1, mouth_encoder_hidden_states.size(1), 1)
                mouth_encoder_hidden_states = mouth_encoder_hidden_states + emotion_embeddings
            
            mouth_decoder_hidden_states, _ = self.mouth_decoder(mouth_encoder_hidden_states, padding_mask=padding_mask)

        # eye decoder
        eye_decoder_hidden_states = None
        if self.training_config.learn_eye:
            eye_encoder_hidden_states = encoder_hidden_states
            eye_decoder_hidden_states, _ = self.eye_decoder(eye_encoder_hidden_states, padding_mask=padding_mask)

        return Model1Output(
            encoder_hidden_states=encoder_hidden_states,
            mouth_decoder_hidden_states=mouth_decoder_hidden_states,
            eye_decoder_hidden_states=eye_decoder_hidden_states
        )


    def train_step(self, batch):
        """
        origin
        collated_batch = {
            "idx": torch.LongTensor([s['idx'] for s in batch]),
            "net_input": {
                "audio_features": collated_features,
                "padding_mask": padding_mask,
            },
            "label_lengths": lengths,
            "label_ntokens": ntokens,
            "ctrl_labels": collated_ctrl_labels
        }

        -> 

        # 这里需要在实际的 train step 前根据 lumi05 mouth indices 从实际的维度为 2007 的向量中进行提取；
        collated_batch = {
            "idx": torch.LongTensor([s['idx'] for s in batch]),
            "net_input": {
                "audio_features": collated_features,
                "padding_mask": padding_mask,
            },
            "label_lengths": lengths,
            "label_ntokens": ntokens,
            "mouth_ctrl_labels": collated_mouth_ctrl_labels,
            "eye_ctrl_labels": collated_eye_ctrl_labels,
            "ctrl_labels": collated_ctrl_labels
        }

        """
        
        model_output = self(batch)

        padding_mask = batch["net_input"]["padding_mask"]  # padding mask 在两个位置起作用，一是实际的模型的 forward 的时候，例如计算 attention 的时候；二是在针对每个位置计算 loss 的时候；第二点通常也可以通过 length 实现；

        loss = 0.
        # mouth loss
        mouth_wing_loss = None
        mouth_l1_loss = None
        if self.training_config.learn_mouth:
            muoth_decoder_hidden_states = model_output.mouth_decoder_hidden_states
            mouth_ctrl_pred = self.mouth_head(muoth_decoder_hidden_states)
            mouth_ctrl_labels = batch["mouth_ctrl_labels"]

            if self.training_config.loss_config.use_wing_loss:
                mouth_wing_loss, mouth_wing_record_loss, mouth_wing_record_num = self.wing_loss_fn(mouth_ctrl_pred, mouth_ctrl_labels, padding_mask)
                loss += self.training_config.loss_config.wing_loss_weight * mouth_wing_loss
            
            if self.training_config.loss_config.use_l1_loss:
                mouth_l1_loss, mouth_l1_record_loss, mouth_l1_record_num  = self.l1_loss_fn(mouth_ctrl_pred, mouth_ctrl_labels, padding_mask)
                loss += self.training_config.loss_config.l1_loss_weight * mouth_l1_loss

            # print(loss)

        # TODO 因为现在还没有跑出 mouth 的基本的效果，因此这里先不管 eye，等到之后把 mouth 调通之后再来重新修改 eye 的代码；
        # eye loss
        if self.training_config.learn_eye:
            eye_decoder_hidden_states = model_output.eye_decoder_hidden_states
            eye_ctrl_pred = self.eye_head(eye_decoder_hidden_states)
            eye_ctrl_labels = batch["eye_ctrl_labels"]
            eye_wing_loss, eye_wing_record_loss, eye_wing_record_num = self.wing_loss_fn(eye_ctrl_pred, eye_ctrl_labels, padding_mask)
            loss += eye_wing_loss

        # TODO 目前保持跟 eege 一样的模型架构，之后可以直接尝试下直接添加一个单独的 decoder 和 head；
        pca_l1_loss = None
        if self.training_config.pca_config.learn_pca:
            ctrl_labels_pca = batch["ctrl_labels"]
            if self.training_config.learn_mouth:
                ctrl_labels_pca[..., :self.config.n_lumi_mouth_channels] = mouth_ctrl_pred
            if self.training_config.learn_eye:
                ctrl_labels_pca[..., self.config.n_lumi_mouth_channels:] = eye_ctrl_pred
            
            pca_pred = self.pca_head(ctrl_labels_pca)
            pca_labels = batch["pca_labels"]
            pca_l1_loss, pca_l1_record_loss, pca_l1_record_num = self.l1_loss_fn(pca_pred, pca_labels, mask=padding_mask)
            loss += pca_l1_loss


        return {
            "loss": loss,
            "mouth_wing_loss_record": (mouth_wing_record_loss, mouth_wing_record_num) if mouth_wing_loss else None,
            "mouth_l1_loss_record": (mouth_l1_record_loss, mouth_l1_record_num) if mouth_l1_loss else None,
            "eye_wing_loss_record": (eye_wing_record_loss, eye_wing_record_num) if self.training_config.learn_eye else None,

            "pca_l1_loss_record": (pca_l1_record_loss, pca_l1_record_num) if pca_l1_loss else None
        }
        

    # 这里对应 evaluate_1.py，即实际上为直接生成使用，因此不写成 evaluate_step;
    def inference_step(self, batch):
        # 虽然默认在调用时是一次一个样本输入，但是这里还是默认接受并且返回 batch 的形式，因此实际上的输入和输出形式基本上还是和 train_step 一样的；

        model_output = self(batch)

        mouth_ctrl_pred = None
        if self.training_config.learn_mouth:
            mouth_ctrl_pred = self.mouth_head(model_output.mouth_decoder_hidden_states)
        eye_ctrl_pred = None
        if self.training_config.learn_eye:
            eye_ctrl_pred = self.eye_head(model_output.eye_decoder_hidden_states)

        return {
            "mouth_ctrl_pred": mouth_ctrl_pred,
            "eye_ctrl_pred": eye_ctrl_pred
        }




















