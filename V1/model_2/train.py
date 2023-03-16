

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


from data_prepare import (
    RawAudioDataset,
    RawAudioDatasetConfig,
    RawAudioCollater,

    ConstantTokenBatchSampler,
    PhnDataset,
    CombinedFeatureDataset,
    NewFeatureDataset
)

from model_2 import (
    Model2Config,
    Model2PL
)

from losses import LossConfig

from helper_fns import parse_config_from_yaml







# TODO 支持多个 checkpoint 的 config；
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
class PhnConfig:
    add_phn_embedding: bool = False
    phn_num: Optional[int] = 63
    phn_layer: Optional[int] = None 

    phn_feature_dir: Optional[str] = None
    phn_padding_idx: Optional[int] = 0
    phn_directly_pad: bool = True
    phn_flatten_pad: bool = True
    


# 和 phn embedding config 一样；
@dataclass
class PcaConfig:
    """
    根据输出的标签控制器的值来预测 pca 值；

    这里需要有一个特殊处理的地方在于，如果没有学 mouth 或者 eye，那么对应部分需要使用 golden label 来进行替换；
    
    """

    learn_pca: bool = False
    pca_label_dir: Optional[str] = None
    n_pca_channels: int = 134


@dataclass
class EmotionConfig:
    """
    该 config 包含 emotion embedding config 和 emotion prediction task config；
    
    """

    add_emotion_embedding: bool = False
    emotion_num: int = 18  #  17 + pad(0)
    emotion_layer: Optional[int] = None

    emotion_feature_dir: Optional[str] = None


@dataclass
class CasualMaskConfig:
    add_double_casual_mask: Optional[bool] = False
    dcm_ratios: Optional[List] = None  # List[float]


@dataclass
class TrainingConfig:

    
    load_checkpoint_path: Optional[str] = None
    resume_training: bool = False

    use_constant_batch_sampler: bool = False

    batch_size: int = 8
    num_workers: int = 32
    
    seed: int = 0
    shuffle: bool = True

    learn_mouth: bool = True
    learn_eye: bool = False

    base_lr: float = 5e-4
    encoder_lr: float = 5e-4  # 用户在配置 encoder optim params 的时候应该使用 pipeline config；
    mouth_head_lr: float = 5e-4
    eye_head_lr: float = 5e-4
    min_lr: float = 1e-5  # TODO 实现控制所有 lr scheduler，现在只控制 LambdaLR；

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

    use_wandb: bool = False
    
    loss_config: LossConfig = LossConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()

    phn_config: PhnConfig = PhnConfig()
    pca_config: PcaConfig = PcaConfig()
    emotion_config: EmotionConfig = EmotionConfig()
    casual_mask_config: CasualMaskConfig = CasualMaskConfig()

    def __post_init__(self):
        assert self.warmup_epochs is None or self.warmup_steps is None



@dataclass
class UsedDatasetConfig:
    train_dataset_config: RawAudioDatasetConfig = None  # 如果这里是 dataclass config，那么这里不能加 Optional;
    validate_dataset_config: RawAudioDatasetConfig = None


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    args = parser.parse_args()

    config_dict = parse_config_from_yaml(args.config_path, UsedDatasetConfig, Model2Config, TrainingConfig)
    used_dataset_config: UsedDatasetConfig = config_dict["UsedDatasetConfig"]
    train_raw_audio_dataset_config: RawAudioDatasetConfig = used_dataset_config.train_dataset_config
    validate_raw_audio_dataset_config: RawAudioDatasetConfig = used_dataset_config.validate_dataset_config
    model1_config: Model2Config = config_dict['Model2Config']
    training_config: TrainingConfig = config_dict['TrainingConfig']

    for key, value in config_dict.items():
        logger.info(f"{key}:\n{value}")

    pl.seed_everything(training_config.seed)

    train_raw_audio_dataset = RawAudioDataset(
        name_manifest_path=train_raw_audio_dataset_config.name_manifest_path,
        audio_dir=train_raw_audio_dataset_config.audio_dir,
        ctrl_label_dir=train_raw_audio_dataset_config.ctrl_label_dir,
        max_keep_sample_size=train_raw_audio_dataset_config.max_keep_sample_size,
        min_keep_sample_size=train_raw_audio_dataset_config.min_keep_sample_size
    )

    train_phn_dataset = None
    phn_config = training_config.phn_config
    if phn_config.add_phn_embedding:
        train_phn_dataset = PhnDataset(name_manifest_path=train_raw_audio_dataset_config.name_manifest_path, phn_dir=phn_config.phn_feature_dir)
    train_pca_dataset = None
    pca_config = training_config.pca_config
    if pca_config.learn_pca:
        train_pca_dataset = NewFeatureDataset(name_manifest_path=train_raw_audio_dataset_config.name_manifest_path, feature_dir=pca_config.pca_label_dir, feature_name="pca_label")
    train_emotion_dataset = None
    emotion_config = training_config.emotion_config
    if emotion_config.add_emotion_embedding:
        train_emotion_dataset = NewFeatureDataset(name_manifest_path=train_raw_audio_dataset_config.name_manifest_path, feature_dir=emotion_config.emotion_feature_dir, feature_name="emotion_id")
    
    train_raw_audio_dataset = CombinedFeatureDataset(train_raw_audio_dataset, train_phn_dataset, train_pca_dataset, train_emotion_dataset, post_process_fn=None)

    train_collate_fn = RawAudioCollater(
        max_sample_size=train_raw_audio_dataset_config.max_sample_size,
        pad_audio=train_raw_audio_dataset_config.pad_audio,
        random_crop=train_raw_audio_dataset_config.random_crop,
        
        sample_rate=train_raw_audio_dataset_config.sample_rate,
        label_rate=train_raw_audio_dataset_config.label_rate,

        phn_directly_pad=phn_config.phn_directly_pad,
        phn_flatten_pad=phn_config.phn_flatten_pad,
        phn_padding_idx=phn_config.phn_padding_idx
    )

    if training_config.use_constant_batch_sampler:
        train_batch_sampler = ConstantTokenBatchSampler(
            size_list=train_raw_audio_dataset.size_list(),
            one_batch_total_tokens=training_config.one_batch_total_tokens,
            shuffle=training_config.shuffle,
            num_buckets=5,
            seed=training_config.seed,
            dataset_name="static feature dataset",
            num_replicas=len(training_config.devices),
            rank=int(os.environ.get("LOCAL_RANK", 0)),
            drop_last=True
        )
        train_dataloader = DataLoader(dataset=train_raw_audio_dataset, batch_sampler=train_batch_sampler, collate_fn=train_collate_fn, num_workers=training_config.num_workers)
    else:
        train_dataloader = DataLoader(dataset=train_raw_audio_dataset, batch_size=training_config.batch_size, shuffle=training_config.shuffle, collate_fn=train_collate_fn,
                                      num_workers=training_config.num_workers)

    validate_raw_audio_dataset = RawAudioDataset(
        name_manifest_path=validate_raw_audio_dataset_config.name_manifest_path,
        audio_dir=validate_raw_audio_dataset_config.audio_dir,
        ctrl_label_dir=validate_raw_audio_dataset_config.ctrl_label_dir,
        max_keep_sample_size=validate_raw_audio_dataset_config.max_keep_sample_size,
        min_keep_sample_size=validate_raw_audio_dataset_config.min_keep_sample_size
    )
    validate_phn_dataset = None
    if phn_config.add_phn_embedding:
        validate_phn_dataset = PhnDataset(name_manifest_path=validate_raw_audio_dataset_config.name_manifest_path, phn_dir=phn_config.phn_feature_dir)
    validate_pca_dataset = None
    if pca_config.learn_pca:
        validate_pca_dataset = NewFeatureDataset(name_manifest_path=validate_raw_audio_dataset_config.name_manifest_path, feature_dir=pca_config.pca_label_dir, feature_name="pca_label")
    validate_emotion_dataset = None
    if emotion_config.add_emotion_embedding:
        validate_emotion_dataset = NewFeatureDataset(name_manifest_path=validate_raw_audio_dataset_config.name_manifest_path, feature_dir=emotion_config.emotion_feature_dir, feature_name="emotion_id")
    
    validate_raw_audio_dataset = CombinedFeatureDataset(validate_raw_audio_dataset, validate_phn_dataset, validate_pca_dataset, validate_emotion_dataset, post_process_fn=None)
    
    validate_dataloader = DataLoader(dataset=validate_raw_audio_dataset, batch_size=training_config.batch_size, shuffle=False, collate_fn=train_collate_fn, num_workers=training_config.num_workers)

    model1_pl = Model2PL(model1_config, training_config)

    if training_config.load_checkpoint_path is not None and not training_config.resume_training:
        model1_pl.load_from_checkpoint(training_config.load_checkpoint_path)

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
        strategy="ddp" if len(training_config.devices) > 1 else None,
        devices=training_config.devices,
        max_epochs=training_config.n_epochs,
        callbacks=callbacks,
        val_check_interval=training_config.val_check_interval,
        check_val_every_n_epoch=training_config.check_val_every_n_epoch,
        gradient_clip_val=training_config.gradient_clip_val
    )

    trainer.fit(
        model=model1_pl,
        train_dataloaders=train_dataloader,
        val_dataloaders=validate_dataloader,
        ckpt_path=training_config.load_checkpoint_path if training_config.resume_training else None
    )




if __name__ == "__main__":


    train()



































