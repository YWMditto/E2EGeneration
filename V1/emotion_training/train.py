



import os
import sys
sys.path.append(".")
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from typing import List, Optional
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import (
    EmotionDataset,
    EmotionDatasetConfig,
    EmotionCollater
)

from model import (
    EmotionClassifier,
    EmotionClassifierConfig
)

from pl_module import EmotionCLassfierPL

from helper_fns import parse_config_from_yaml


@dataclass
class CheckpointConfig:
    save_checkpoint: bool = False
    checkpoint_dir: Optional[str] = None
    checkpoint_sub_name: Optional[str] = None

    monitor: Optional[str] = None
    save_last: bool = True
    save_top_k: int = 1
    save_weights_only: bool = False
    mode: str = "min"
    every_n_epochs: Optional[int] = 1
    every_n_train_steps: Optional[int] = None
    save_on_train_epoch_end: Optional[bool] = False


@dataclass
class TrainingConfig:

    load_checkpoint_path: Optional[str] = None
    resume_training: bool = False

    batch_size: int = 8
    num_workers: int = 32
    
    seed: int = 0
    shuffle: bool = True

    base_lr: float = 5e-4
    weight_decay: float = 0

    lr_T_0: int = 20
    lr_T_mult: int = 1
    lr_eta_min: float = 1e-6

    warmup_epochs: Optional[int] = None
    warmup_steps: Optional[int] = None
    n_epochs: int = 100

    val_check_interval: Optional[int] = None
    check_val_every_n_epoch: int = 1

    gradient_accumulation_step: int = 1
    gradient_clip_val: Optional[float] = None

    devices: List[int] = field(default_factory=lambda: [0])

    checkpoint_config: CheckpointConfig = CheckpointConfig()


@dataclass
class UsedDatasetConfig:
    train_dataset_config: EmotionDatasetConfig = None  # 如果这里是 dataclass config，那么这里不能加 Optional;
    validate_dataset_config: EmotionDatasetConfig = None


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    args = parser.parse_args()

    config_dict = parse_config_from_yaml(args.config_path, UsedDatasetConfig, EmotionClassifierConfig, TrainingConfig)
    used_dataset_config: UsedDatasetConfig = config_dict["UsedDatasetConfig"]
    train_emotion_dataset_config: EmotionDatasetConfig = used_dataset_config.train_dataset_config
    validate_emotion_dataset_config: EmotionDatasetConfig = used_dataset_config.validate_dataset_config
    model_config: EmotionClassifierConfig = config_dict['EmotionClassifierConfig']
    training_config: TrainingConfig = config_dict['TrainingConfig']

    for key, value in config_dict.items():
        logger.info(f"{key}:\n{value}")

    pl.seed_everything(training_config.seed)

    train_emotion_dataset = EmotionDataset(
        name_manifest_path=train_emotion_dataset_config.name_manifest_path,
        ctrl_label_dir=train_emotion_dataset_config.ctrl_label_dir,
        emotion_label_dir=train_emotion_dataset_config.emotion_label_dir
    )
    
    train_collate_fn = EmotionCollater()

    train_dataloader = DataLoader(dataset=train_emotion_dataset, batch_size=training_config.batch_size, shuffle=training_config.shuffle, collate_fn=train_collate_fn,
                                    num_workers=training_config.num_workers)

    validate_raw_audio_dataset = EmotionDataset(
        name_manifest_path=validate_emotion_dataset_config.name_manifest_path,
        ctrl_label_dir=validate_emotion_dataset_config.ctrl_label_dir,
        emotion_label_dir=validate_emotion_dataset_config.emotion_label_dir
    )
    
    validate_dataloader = DataLoader(dataset=validate_raw_audio_dataset, batch_size=training_config.batch_size, shuffle=False, collate_fn=train_collate_fn, num_workers=training_config.num_workers)

    model_pl = EmotionCLassfierPL(model_config, training_config)

    if training_config.load_checkpoint_path is not None and not training_config.resume_training:
        model_pl.load_from_checkpoint(training_config.load_checkpoint_path)

    callbacks = []

    if training_config.checkpoint_config.save_checkpoint:
        checkpoint_config = training_config.checkpoint_config
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_config.checkpoint_dir,
            filename=checkpoint_config.checkpoint_sub_name,
            monitor=checkpoint_config.monitor,
            save_last=checkpoint_config.save_last,
            save_top_k=checkpoint_config.save_top_k,
            mode=checkpoint_config.mode,
            auto_insert_metric_name=True,
            save_weights_only=checkpoint_config.save_weights_only,
            every_n_epochs=checkpoint_config.every_n_epochs,
            every_n_train_steps=checkpoint_config.every_n_train_steps,
            save_on_train_epoch_end=checkpoint_config.save_on_train_epoch_end
        )
        callbacks.append(checkpoint_callback)


    trainer = pl.Trainer(
        accelerator="gpu",
        devices=training_config.devices,
        max_epochs=training_config.n_epochs,
        callbacks=callbacks,
        val_check_interval=training_config.val_check_interval,
        check_val_every_n_epoch=training_config.check_val_every_n_epoch,
        gradient_clip_val=training_config.gradient_clip_val
    )

    trainer.fit(
        model=model_pl,
        train_dataloaders=train_dataloader,
        val_dataloaders=[validate_dataloader, train_dataloader],
        ckpt_path=training_config.load_checkpoint_path if training_config.resume_training else None
    )


if __name__ == "__main__":
    train()








































