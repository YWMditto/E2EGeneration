



import torch
import pytorch_lightning as pl
from itertools import chain

try:
    import wandb
except:
    ...

import logging
logger = logging.getLogger(__file__)

from .models import Model1, Model1Config
from data_prepare import StaticFeatureDatasetConfig




class Model1PL(pl.LightningModule):
    def __init__(self, model1_config: Model1Config, training_config: "TrainingConfig", static_feature_dataset_config: StaticFeatureDatasetConfig) -> None:
        super().__init__()

        self.model1_config = model1_config
        self.training_config = training_config
        self.static_feature_dataset_config = static_feature_dataset_config

        self.model1 = Model1(model1_config, training_config)
        # self.lumi05_mouth_ctrl_indices = static_feature_dataset_config.lumi05_mouth_without_R_ctrl_indices
        # self.lumi05_eye_ctrl_indices = static_feature_dataset_config.lumi05_eye_without_R_ctrl_indices
        self.lumi05_mouth_ctrl_indices_num = len(static_feature_dataset_config.lumi05_mouth_without_R_ctrl_indices)

        self.use_wandb = training_config.use_wandb

    # # past
    # def training_step(self, batch, batch_idx):
    #     collated_ctrl_labels = batch["ctrl_labels"]
    #     mouth_ctrl_labels = collated_ctrl_labels[..., self.lumi05_mouth_ctrl_indices]
    #     eye_ctrl_labels = collated_ctrl_labels[..., self.lumi05_eye_ctrl_indices]
    #     batch["mouth_ctrl_labels"] = mouth_ctrl_labels
    #     batch["eye_ctrl_labels"] = eye_ctrl_labels
    #     loss_dict = self.model1.train_step(batch)
    #     loss = loss_dict["loss"]
        
    #     loss_dict.pop("loss")
    #     loss_dict["global_step"] = float(self.global_step)

    #     mouth_wing_loss_record = loss_dict.pop("mouth_wing_loss_record")
    #     loss_dict["mouth_wing_loss"] = mouth_wing_loss_record[0] / mouth_wing_loss_record[1]
    #     eye_wing_loss_record = loss_dict.pop("eye_wing_loss_record")
    #     loss_dict["eye_wing_loss"] = eye_wing_loss_record[0] / eye_wing_loss_record[1]
        
    #     if self.use_wandb:
    #         loss_dict["loss"] = loss.item()
    #         wandb.log(loss_dict)
    #     else:
    #         self.log_dict(loss_dict, prog_bar=True)

    #     # self.log("bsz", float(len(collated_ctrl_labels)), prog_bar=True)
    #     # self.log("feature_len", float(collated_ctrl_labels.size(1)), prog_bar=True)

    #     # TODO 实现一下记录 optimizer lr 的功能；
    #     return loss


    # 使用 normalized_extracted_ctrl_labels 的数据；
    def training_step(self, batch, batch_idx):
        collated_ctrl_labels = batch["ctrl_labels"]
        mouth_ctrl_labels = collated_ctrl_labels[..., :self.lumi05_mouth_ctrl_indices_num]
        eye_ctrl_labels = collated_ctrl_labels[..., self.lumi05_mouth_ctrl_indices_num:]
        batch["mouth_ctrl_labels"] = mouth_ctrl_labels
        batch["eye_ctrl_labels"] = eye_ctrl_labels
        loss_dict = self.model1.train_step(batch)
        loss = loss_dict["loss"]
        
        loss_dict.pop("loss")
        loss_dict["global_step"] = float(self.global_step)

        mouth_wing_loss_record = loss_dict.pop("mouth_wing_loss_record")
        loss_dict["mouth_wing_loss"] = mouth_wing_loss_record[0] / mouth_wing_loss_record[1]
        eye_wing_loss_record = loss_dict.pop("eye_wing_loss_record")
        loss_dict["eye_wing_loss"] = eye_wing_loss_record[0] / eye_wing_loss_record[1]
        
        if self.use_wandb:
            loss_dict["loss"] = loss.item()
            wandb.log(loss_dict)
        else:
            self.log_dict(loss_dict, prog_bar=True)

        # self.log("bsz", float(len(collated_ctrl_labels)), prog_bar=True)
        # self.log("feature_len", float(collated_ctrl_labels.size(1)), prog_bar=True)

        # TODO 实现一下记录 optimizer lr 的功能；
        return loss
    
    def on_after_backward(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.model1.mouth_head.parameters(), 1.0)#self.training_config.gradient_clip_val)
        torch.nn.utils.clip_grad_norm_(self.model1.eye_head.parameters(), 1.0)#self.training_config.gradient_clip_val)

    # # past
    # def validation_step(self, batch, batch_idx):
    #     collated_ctrl_labels = batch["ctrl_labels"]
    #     mouth_ctrl_labels = collated_ctrl_labels[..., self.lumi05_mouth_ctrl_indices]
    #     eye_ctrl_labels = collated_ctrl_labels[..., self.lumi05_eye_ctrl_indices]
    #     batch["mouth_ctrl_labels"] = mouth_ctrl_labels
    #     batch["eye_ctrl_labels"] = eye_ctrl_labels
    #     loss_dict = self.model1.train_step(batch)
    #     mouth_wing_loss_record = loss_dict["mouth_wing_loss_record"]
    #     eye_wing_loss_record = loss_dict["eye_wing_loss_record"]
    #     return [mouth_wing_loss_record, eye_wing_loss_record]

    def validation_step(self, batch, batch_idx):
        collated_ctrl_labels = batch["ctrl_labels"]
        mouth_ctrl_labels = collated_ctrl_labels[..., :self.lumi05_mouth_ctrl_indices_num]
        eye_ctrl_labels = collated_ctrl_labels[..., self.lumi05_mouth_ctrl_indices_num:]
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

        loss_dict = {"validate_loss": mouth_wing_validate_loss + eye_wing_validate_loss, "mouth_wing_validate_loss": mouth_wing_validate_loss, "eye_wing_validate_loss": eye_wing_validate_loss}
        if self.use_wandb:
            wandb.log(loss_dict)
        else:
            self.log_dict(loss_dict, prog_bar=False)
        logger.info(f"validate: mouth wing / eye wing validate loss: {mouth_wing_validate_loss} / {eye_wing_validate_loss}.")

    def on_before_optimizer_step(self, optimizer, optimizer_idx: int) -> None:
        a = 1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model1.parameters(),
            lr=self.training_config.base_lr,
            weight_decay=self.training_config.weight_decay
        )

        lr_scheduler = [
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.training_config.epoch_milestones,
                gamma=self.training_config.gamma,
                last_epoch=self.current_epoch-1  # TODO 断点重训的行为需要测试； 
            )
        ]
        # warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer=optimizer,
        #     lr_lambda=lambda epoch: max(epoch / self.training_config.warmup_epochs, self.training_config.min_lr) ,
        #     last_epoch=self.current_epoch-1
        # )
        # lr_scheduler.append(warmup_scheduler)

        return [optimizer], lr_scheduler


    # TODO 在实现更复杂的 config 设置后，实现 on_save_checkpoint 方法来保存训练过程当中使用的 dataclass config，现在不需要；        


