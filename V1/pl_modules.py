



import torch
import pytorch_lightning as pl
from itertools import chain


from models import Model1, Model1Config
from data_prepare import StaticFeatureDatasetConfig




class Model1PL(pl.LightningModule):
    def __init__(self, model1_config: Model1Config, training_config: "TrainingConfig", static_feature_dataset_config: StaticFeatureDatasetConfig) -> None:
        super().__init__()

        self.model1_config = model1_config
        self.training_config = training_config
        self.static_feature_dataset_config = static_feature_dataset_config

        self.model1 = Model1(model1_config, training_config)
        self.lumi05_mouth_ctrl_indices = static_feature_dataset_config.lumi05_mouth_ctrl_indices
        self.lumi05_eye_ctrl_indices = static_feature_dataset_config.lumi05_eye_ctrl_indices

    def training_step(self, batch, batch_idx):
        collated_ctrl_labels = batch["collated_ctrl_labels"]
        mouth_ctrl_labels = collated_ctrl_labels[..., self.lumi05_mouth_ctrl_indices]
        eye_ctrl_labels = collated_ctrl_labels[..., self.lumi05_eye_ctrl_indices]
        batch["mouth_ctrl_labels"] = mouth_ctrl_labels
        batch["eye_ctrl_labels"] = eye_ctrl_labels
        loss_dict = self.model1.train_step(batch)
        return loss_dict["loss"]
    
    # TODO 实现 validate step 以及 validate metric；

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=[
                {
                    "params": self.model1.encoder.parameters(),
                    "lr": self.training_config.encoder_lr,
                    "weight_decay": self.training_config.weight_decay
                },
                {
                    "params": self.model1.decoder.parameters(),
                    "lr": self.training_config.decoder_lr,
                    "weight_decay": self.training_config.weight_decay
                },
                {
                    "params": self.model1.pre_proj.parameters(),
                    "lr": self.training_config.pre_proj_lr,
                    "weight_decay": self.training_config.weight_decay
                },
                {
                    "params": self.model1.mouth_head.parameters(),
                    "lr": self.training_config.mouth_head_lr,
                    "weight_decay": self.training_config.weight_decay
                },
                {
                    "params": self.model1.eye_head.parameters(),
                    "lr": self.training_config.eye_head_lr,
                    "weight_decay": self.training_config.weight_decay
                },
            ],
            lr=self.training_config.base_lr,
            weight_decay=self.training_config.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.training_config.epoch_milestones,
            gamma=self.training_config.gamma,

            # TODO 完成断点重训后在这里添加 last_epoch；
        )
        return [optimizer], [lr_scheduler]






