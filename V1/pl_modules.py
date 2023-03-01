



import torch
import pytorch_lightning as pl
from itertools import chain

try:
    import wandb
except:
    ...

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

        self.use_wandb = training_config.use_wandb


    def training_step(self, batch, batch_idx):
        collated_ctrl_labels = batch["ctrl_labels"]
        mouth_ctrl_labels = collated_ctrl_labels[..., self.lumi05_mouth_ctrl_indices]
        eye_ctrl_labels = collated_ctrl_labels[..., self.lumi05_eye_ctrl_indices]
        batch["mouth_ctrl_labels"] = mouth_ctrl_labels
        batch["eye_ctrl_labels"] = eye_ctrl_labels
        loss_dict = self.model1.train_step(batch)
        loss = loss_dict["loss"]
        
        loss_dict.pop("loss")
        loss_dict["global_step"] = float(self.global_step)
        loss_dict["loss"] = loss.item()

        mouth_wing_loss_record = loss_dict.pop("mouth_wing_loss_record")
        loss_dict["mouth_wing_loss"] = mouth_wing_loss_record[0] / mouth_wing_loss_record[1]
        eye_wing_loss_record = loss_dict.pop("eye_wing_loss_record")
        loss_dict["eye_wing_loss"] = eye_wing_loss_record[0] / eye_wing_loss_record[1]
        
        if self.use_wandb:
            wandb.log(loss_dict)
        else:
            self.log_dict(loss_dict)

        self.log("bsz", float(len(collated_ctrl_labels)), prog_bar=True)
        self.log("feature_len", float(collated_ctrl_labels.size(1)), prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        collated_ctrl_labels = batch["collated_ctrl_labels"]
        mouth_ctrl_labels = collated_ctrl_labels[..., self.lumi05_mouth_ctrl_indices]
        eye_ctrl_labels = collated_ctrl_labels[..., self.lumi05_eye_ctrl_indices]
        batch["mouth_ctrl_labels"] = mouth_ctrl_labels
        batch["eye_ctrl_labels"] = eye_ctrl_labels
        loss_dict = self.model1.train_step(batch)
        mouth_wing_loss_record = loss_dict["mouth_wing_loss_record"]
        eye_wing_loss_record = loss_dict["eye_wing_loss_record"]
        return [mouth_wing_loss_record, eye_wing_loss_record]
    
    def validation_epoch_end(self, outputs) -> None:
        mouth_wing_loss_records, eye_wing_loss_records = list(zip(*outputs))
        mouth_wing_losses, mouth_wing_loss_num = list(zip(*mouth_wing_loss_records))
        eye_wing_losses, eye_wing_loss_num = list(zip(*eye_wing_loss_records))
        mouth_wing_validate_loss = sum(mouth_wing_losses) / (sum(mouth_wing_loss_num))
        eye_wing_validate_loss = sum(eye_wing_losses) / (sum(eye_wing_loss_num))

        loss_dict = {"mouth_wing_validate_loss": mouth_wing_validate_loss, "eye_wing_validate_loss": eye_wing_validate_loss}
        if self.use_wandb:
            wandb.log(loss_dict)
        else:
            self.log_dict(loss_dict, prog_bar=True)

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





