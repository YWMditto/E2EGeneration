



import torch
import pytorch_lightning as pl
from itertools import chain

try:
    import wandb
except:
    ...

import logging
logger = logging.getLogger(__file__)

from .models import Model2, Model2Config
from data_prepare import StaticFeatureDatasetConfig


from pytorch_lightning.callbacks.progress import TQDMProgressBar

class Model2PL(pl.LightningModule):
    def __init__(self, model1_config: Model2Config, training_config: "TrainingConfig") -> None:
        super().__init__()

        self.model1_config = model1_config
        self.training_config = training_config

        self.model2 = Model2(model1_config, training_config)

        self.use_wandb = training_config.use_wandb

        self.lr_scale = None


    # 使用 normalized_extracted_ctrl_labels 的数据；
    def training_step(self, batch, batch_idx):
        loss_dict = self.model2.train_step(batch)
        loss = loss_dict["loss"]
        
        loss_dict.pop("loss")
        loss_dict["global_step"] = float(self.global_step)

        mouth_wing_loss_record = loss_dict.pop("mouth_wing_loss_record")
        if mouth_wing_loss_record is not None:
            loss_dict["mouth_wing_loss"] = mouth_wing_loss_record[0] / mouth_wing_loss_record[1]

        mouth_l1_loss_record = loss_dict.pop("mouth_l1_loss_record")
        if mouth_l1_loss_record is not None:
            loss_dict["mouth_l1_loss"] = mouth_l1_loss_record[0] / mouth_l1_loss_record[1]

        eye_wing_loss_record = loss_dict.pop("eye_wing_loss_record", None)
        if eye_wing_loss_record is not None:
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
    
    # TODO 之后试一下加上是不是好一点，通常来说是的；
    # def on_after_backward(self) -> None:
    #     if self.training_config.learn_mouth:
    #         torch.nn.utils.clip_grad_norm_(self.model2.mouth_head.parameters(), 1.0)#self.training_config.gradient_clip_val)
    #     if self.training_config.learn_eye:
    #         torch.nn.utils.clip_grad_norm_(self.model2.eye_head.parameters(), 1.0)#self.training_config.gradient_clip_val)

    def on_train_epoch_start(self) -> None:
        if self.training_config.warmup_epochs is not None:
            self.lr_scale = min(1.0, float(self.current_epoch + 1) / int(self.training_config.warmup_epochs))

    def on_before_optimizer_step(self, optimizer, optimizer_idx: int):
        
        if self.training_config.warmup_steps is not None:
            lr_scale = min(1.0, float(self.global_step + 1) / int(self.training_config.warmup_steps))
        # 由 on epoch 时进行设置；
        else:
            lr_scale = self.lr_scale

        if lr_scale is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scale * self.training_config.base_lr


    def validation_step(self, batch, batch_idx):
        loss_dict = self.model2.train_step(batch)
        mouth_wing_loss_record = loss_dict.get("mouth_wing_loss_record", None)
        mouth_l1_loss_record = loss_dict.get("mouth_l1_loss_record", None)
        eye_wing_loss_record = loss_dict.get("eye_wing_loss_record", None)
        return [mouth_wing_loss_record, eye_wing_loss_record, mouth_l1_loss_record]
    
    def validation_epoch_end(self, outputs) -> None:
        
        mouth_wing_loss_records, eye_wing_loss_records, mouth_l1_loss_records = list(zip(*outputs))

        mouth_wing_validate_loss = 0.
        if mouth_wing_loss_records[0] is not None:
            mouth_wing_losses, mouth_wing_loss_num = list(zip(*mouth_wing_loss_records))
            mouth_wing_validate_loss = sum(mouth_wing_losses) / (sum(mouth_wing_loss_num))
        
        mouth_l1_validate_loss = 0.
        if mouth_l1_loss_records[0] is not None:
            mouth_l1_losses, mouth_l1_loss_num = list(zip(*mouth_l1_loss_records))
            mouth_l1_validate_loss = sum(mouth_l1_losses) / (sum(mouth_l1_loss_num))

        eye_wing_validate_loss = 0.
        if eye_wing_loss_records[0] is not None:
            eye_wing_losses, eye_wing_loss_num = list(zip(*eye_wing_loss_records))
            eye_wing_validate_loss = sum(eye_wing_losses) / (sum(eye_wing_loss_num))

        loss_dict = {"validate_loss": mouth_wing_validate_loss + eye_wing_validate_loss + mouth_l1_validate_loss, "mouth_wing_validate_loss": mouth_wing_validate_loss, "mouth_l1_validate_loss": mouth_l1_validate_loss, "eye_wing_validate_loss": eye_wing_validate_loss}
        if self.use_wandb:
            wandb.log(loss_dict)
        else:
            self.log_dict(loss_dict, prog_bar=False)
        logger.info(f"validate: mouth wing / eye wing validate loss: {mouth_wing_validate_loss} / {eye_wing_validate_loss}, mouth_l1_validate_loss: {mouth_l1_validate_loss}.")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model2.parameters(),
            lr=self.training_config.base_lr,
            weight_decay=self.training_config.weight_decay
        )

        lr_scheduler = [
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=self.training_config.lr_T_0, 
                T_mult=self.training_config.lr_T_mult, 
                eta_min=self.training_config.lr_eta_min,
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


