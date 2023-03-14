









import torch
import pytorch_lightning as pl
from itertools import chain

import logging
logger = logging.getLogger(__file__)

from model import EmotionClassifier, EmotionClassifierConfig


class EmotionCLassfierPL(pl.LightningModule):
    def __init__(self, model_config: EmotionClassifierConfig, training_config: "TrainingConfig") -> None:
        super().__init__()

        self.model_config = model_config
        self.training_config = training_config

        self.model = EmotionClassifier(model_config)

        self.lr_scale = None

    # 使用 normalized_extracted_ctrl_labels 的数据；
    def training_step(self, batch, batch_idx):
        loss_dict = self.model.train_step(batch)
        loss = loss_dict["loss"]
        
        self.log("global_step", float(self.global_step))
        return loss
    
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


    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0:
            dataloader_idx = 0
        else:
            dataloader_idx = args[0]

        returned_dict = self.model.evaluate_step(batch)
        loss = returned_dict["loss"].item()

        preds = returned_dict["preds"]
        emotion_labels = batch["emotion_labels"]
        right_num = torch.sum(preds == emotion_labels).item()
        total_num = len(emotion_labels)

        validate_dict = {"dataloader_idx": dataloader_idx, "loss": loss, "right_num": right_num, "total_num": total_num}
        return validate_dict
    
    def validation_epoch_end(self, outputs) -> None:

        def _f(validate_outupts, prefix="validate_"):
            validate_loss_list = [o["loss"] for o in validate_outupts]
            right_num_list = [o["right_num"] for o in validate_outupts]
            total_num_list = [o["total_num"] for o in validate_outupts]

            validate_loss = sum(validate_loss_list) / len(validate_loss_list)
            acc = sum(right_num_list) / sum(total_num_list)

            print(sum(right_num_list), sum(total_num_list))

            logger_info = f"validate: {prefix}loss: {validate_loss},\t{prefix}acc: {acc}"
            log_dict = {
                f"{prefix}loss": validate_loss,
                f"{prefix}acc": acc
            }

            return log_dict, logger_info
        if isinstance(outputs[0], dict):
            validate_outupts = outputs
        else:
            validate_outupts = [o for o in outputs[0]]
        validate_log_dict, validate_logger_info = _f(validate_outupts)
        self.log_dict(validate_log_dict, prog_bar=False)
        logger.info(validate_logger_info)

        # 评测 train dataloader；
        if not isinstance(outputs[0], dict):
            train_outupts = [o for o in outputs[1]]
            train_log_dict, train_logger_info = _f(train_outupts, "train_")
            self.log_dict(train_log_dict, prog_bar=False)
            logger.info(train_logger_info)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
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
        
        return [optimizer], lr_scheduler


