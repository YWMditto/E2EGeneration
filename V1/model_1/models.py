

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from .modules import EEGTransformer, PhnModel
from losses import WingLoss




def _init_weights(module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight.data)
    if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
        module.bias.data.zero_()



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

    decoder_layer_num: int = 5
    decoder_head_num: int = 1
    decoder_head_dim: int = 64
    decoder_conv1d_filter_size: int = 1536
    decoder_conv1d_kernel_size: int = 3
    decoder_dropout_p: float = 0.1
    decoder_dropatt_p: float = 0.1
    decoder_dropemb_p: float = 0.0


    # phn config 在 training config 中进行设置；



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

        self.mouth_decoder = EEGTransformer(
            n_layer=config.decoder_layer_num, 
            n_head=config.decoder_head_num, 
            d_model=config.hidden_size, 
            d_head=config.decoder_head_dim, 
            d_inner=config.decoder_conv1d_filter_size, 
            kernel_size=config.decoder_conv1d_kernel_size,
            dropout=config.decoder_dropout_p, 
            dropatt=config.decoder_dropatt_p, 
            dropemb=config.decoder_dropemb_p, 
            pre_lnorm=config.pre_lnorm
        )
        self.mouth_head = nn.Linear(in_features=config.hidden_size, out_features=config.n_lumi_mouth_channels)


        self.eye_decoder = EEGTransformer(
            n_layer=config.decoder_layer_num, 
            n_head=config.decoder_head_num, 
            d_model=config.hidden_size, 
            d_head=config.decoder_head_dim, 
            d_inner=config.decoder_conv1d_filter_size, 
            kernel_size=config.decoder_conv1d_kernel_size,
            dropout=config.decoder_dropout_p, 
            dropatt=config.decoder_dropatt_p, 
            dropemb=config.decoder_dropemb_p, 
            pre_lnorm=config.pre_lnorm
        )
        self.eye_head = nn.Linear(in_features=config.hidden_size, out_features=config.n_lumi_eye_channels)

        # 注意如果之后开始使用预训练模型，那么需要注意随机初始化的位置；
        self.apply(_init_weights)

        # phn embedding

        if training_config.phn_embedding_config.add_phn:
            self.phn_model = PhnModel(
                phn_num=training_config.phn_embedding_config.phn_num,
                hidden_size=config.hidden_size,
                padding_idx=training_config.phn_embedding_config.phn_padding_idx
            )


        self.loss_config = training_config.loss_config
        self.wing_loss_config = self.loss_config.wing_loss_config
        self.wing_loss_fn = WingLoss(omega=self.wing_loss_config.omega, epsilon=self.wing_loss_config.epsilon, emoji_weight=self.wing_loss_config.emoji_weight)

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
        mouth_encoder_hidden_states = encoder_hidden_states
        if self.training_config.phn_embedding_config.add_phn:
            phn_dict = batch["phn_dict"]
            phn_hidden_states = self.phn_model(phns=phn_dict["collated_phns"])
            mouth_encoder_hidden_states = mouth_encoder_hidden_states + phn_hidden_states
        mouth_decoder_hidden_states, _ = self.mouth_decoder(mouth_encoder_hidden_states, padding_mask=padding_mask)

        # eye decoder
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
        }

        """
        
        model_output = self(batch)

        padding_mask = batch["net_input"]["padding_mask"]  # padding mask 在两个位置起作用，一是实际的模型的 forward 的时候，例如计算 attention 的时候；二是在针对每个位置计算 loss 的时候；第二点通常也可以通过 length 实现；

        # mouth loss
        muoth_decoder_hidden_states = model_output.mouth_decoder_hidden_states
        mouth_ctrl_pred = self.mouth_head(muoth_decoder_hidden_states)
        mouth_ctrl_labels = batch["mouth_ctrl_labels"]
        mouth_wing_loss, mouth_wing_record_loss, mouth_wing_record_num  = self.wing_loss_fn(mouth_ctrl_pred, mouth_ctrl_labels, padding_mask)
        
        # ete loss
        eye_decoder_hidden_states = model_output.eye_decoder_hidden_states
        eye_ctrl_pred = self.eye_head(eye_decoder_hidden_states)
        eye_ctrl_labels = batch["eye_ctrl_labels"]
        eye_wing_loss, eye_wing_record_loss, eye_wing_record_num = self.wing_loss_fn(eye_ctrl_pred, eye_ctrl_labels, padding_mask)

        loss = mouth_wing_loss + eye_wing_loss

        return {
            "loss": loss,
            "mouth_wing_loss_record": (mouth_wing_record_loss, mouth_wing_record_num),
            "eye_wing_loss_record": (eye_wing_record_loss, eye_wing_record_num)
        }
        

    # 这里对应 evaluate_1.py，即实际上为直接生成使用，因此不写成 evaluate_step;
    def inference_step(self, batch):
        # 虽然默认在调用时是一次一个样本输入，但是这里还是默认接受并且返回 batch 的形式，因此实际上的输入和输出形式基本上还是和 train_step 一样的；

        model_output = self(batch)

        mouth_ctrl_pred = self.mouth_head(model_output.mouth_decoder_hidden_states)
        eye_ctrl_pred = self.eye_head(model_output.eye_decoder_hidden_states)

        return {
            "mouth_ctrl_pred": mouth_ctrl_pred,
            "eye_ctrl_pred": eye_ctrl_pred
        }




















