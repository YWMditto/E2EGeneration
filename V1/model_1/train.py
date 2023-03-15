

"""
TODO 实现对于 Model1 的训练，之后应当从中抽取出共用的 train_utils；
"""
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
    StaticFeatureDatasetConfig,
    StaticFeatureDataset,
    StaticFeatureCollater,
    ConstantTokenBatchSampler,
    PhnDataset,
    CombinedFeatureDataset,
    static_feature_phn_post_proces_fn,
    NewFeatureDataset,
    check_feature_length_post_process_fn
)

from model_1 import (
    Model1Config,
    Model1PL
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


# 将这一 config 写在这里是为了支持便捷地插入和删除这一组件；之后可能会分散到 model_config 和 data_config 中；
@dataclass
class PhnEmbeddingConfig:
    add_phn: bool = False
    phn_dir: Optional[str] = None
    phn_num: Optional[int] = None
    phn_padding_idx: Optional[int] = None
    phn_directly_pad: bool = True
    phn_flatten_pad: bool = True

    phn_layer_num: int = 6
    phn_head_num: int = 1
    phn_head_dim: int = 64
    phn_conv1d_filter_size: int = 1536
    phn_conv1d_kernel_size: int = 3
    # encoder_output_size: 384
    phn_dropout_p: float = 0.1
    phn_dropatt_p: float = 0.1
    phn_dropemb_p: float = 0.0


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
    emotion_feature_dir: Optional[str] = None

    # emotion prediction task;
    learn_emotion: bool = False
    emotion_classify_loss_type: str = "BCELoss"
    emotion_classify_loss_weight: float = 0.1
    emotion_classifier_config_path: Optional[str] = None
    emotion_classifier_ckpt_path: Optional[str] = None





@dataclass
class TrainingConfig:
    
    load_checkpoint_path: Optional[str] = None
    resume_training: bool = False

    use_constant_batch_sampler: bool = False
    one_batch_total_tokens: int = 6000

    batch_size: int = 8
    num_workers: int = 32
    
    seed: int = 0
    shuffle: bool = True

    base_lr: float = 5e-4
    pre_proj_lr: float = 5e-4
    encoder_lr: float = 5e-4  # 用户在配置 encoder optim params 的时候应该使用 pipeline config；
    decoder_lr: float = 5e-4
    mouth_head_lr: float = 5e-4
    eye_head_lr: float = 5e-4
    min_lr: float = 1e-5  # TODO 实现控制所有 lr scheduler，现在只控制 LambdaLR；

    learn_mouth: bool = True
    learn_eye: bool = False
    learn_pca: bool = False

    weight_decay: float = 1e-5
    epoch_milestones: List[int] = field(default_factory=lambda: [12, 40, 70])
    gamma: float = 0.2

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

    phn_embedding_config: PhnEmbeddingConfig = PhnEmbeddingConfig()
    pca_config: PcaConfig = PcaConfig()
    emotion_config: EmotionConfig = EmotionConfig()

    def __post_init__(self):
        assert self.warmup_epochs is None or self.warmup_steps is None


@dataclass
class UsedDatasetConfig:
    train_dataset_config: StaticFeatureDatasetConfig = None  # 如果这里是 dataclass config，那么这里不能加 Optional;
    validate_dataset_config: StaticFeatureDatasetConfig = None


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    args = parser.parse_args()

    config_dict = parse_config_from_yaml(args.config_path, UsedDatasetConfig, Model1Config, TrainingConfig)
    used_dataset_config: UsedDatasetConfig = config_dict["UsedDatasetConfig"]
    train_static_feature_dataset_config: StaticFeatureDatasetConfig = used_dataset_config.train_dataset_config
    validate_static_feature_dataset_config: StaticFeatureDatasetConfig = used_dataset_config.validate_dataset_config
    model1_config: Model1Config = config_dict['Model1Config']
    training_config: TrainingConfig = config_dict['TrainingConfig']
    

    for key, value in config_dict.items():
        logger.info(f"{key}:\n{value}")

    pl.seed_everything(training_config.seed)

    train_static_feature_dataset = StaticFeatureDataset(
        name_manifest_path=train_static_feature_dataset_config.name_manifest_path,
        static_feature_dir=train_static_feature_dataset_config.static_feature_dir,
        ctrl_label_dir=train_static_feature_dataset_config.ctrl_label_dir,
        max_keep_feature_size=train_static_feature_dataset_config.max_keep_feature_size,
        min_keep_feature_size=train_static_feature_dataset_config.min_keep_feature_size
    )
    train_phn_dataset = None
    phn_embedding_config = training_config.phn_embedding_config
    if phn_embedding_config.add_phn:
        train_phn_dataset = PhnDataset(name_manifest_path=train_static_feature_dataset_config.name_manifest_path, phn_dir=phn_embedding_config.phn_dir)
    train_pca_dataset = None
    pca_config = training_config.pca_config
    if pca_config.learn_pca:
        train_pca_dataset = NewFeatureDataset(name_manifest_path=train_static_feature_dataset_config.name_manifest_path, feature_dir=pca_config.pca_label_dir, feature_name="pca_label")
    
    # TODO emotoin dataset 的形式之后可能会大改；
    train_emotion_dataset = None
    emotion_config = training_config.emotion_config
    if emotion_config.add_emotion_embedding:
        train_emotion_dataset = NewFeatureDataset(name_manifest_path=train_static_feature_dataset_config.name_manifest_path, feature_dir=emotion_config.emotion_feature_dir, feature_name="emotion_index")

    train_static_feature_dataset = CombinedFeatureDataset(train_static_feature_dataset, train_phn_dataset, train_pca_dataset, train_emotion_dataset, post_process_fn=check_feature_length_post_process_fn)

    train_collate_fn = StaticFeatureCollater(
        max_feature_size=train_static_feature_dataset_config.max_feature_size,
        pad_feature=train_static_feature_dataset_config.pad_feature,
        random_crop=train_static_feature_dataset_config.random_crop,

        phn_directly_pad=phn_embedding_config.phn_directly_pad,
        phn_flatten_pad=phn_embedding_config.phn_flatten_pad,
        phn_padding_idx=phn_embedding_config.phn_padding_idx
    )

    train_dataloader = DataLoader(dataset=train_static_feature_dataset, batch_size=training_config.batch_size, shuffle=training_config.shuffle, collate_fn=train_collate_fn,
                                    num_workers=training_config.num_workers)

    validate_static_feature_dataset = StaticFeatureDataset(
        name_manifest_path=validate_static_feature_dataset_config.name_manifest_path,
        static_feature_dir=validate_static_feature_dataset_config.static_feature_dir,
        ctrl_label_dir=validate_static_feature_dataset_config.ctrl_label_dir,
        max_keep_feature_size=validate_static_feature_dataset_config.max_keep_feature_size,
        min_keep_feature_size=validate_static_feature_dataset_config.min_keep_feature_size
    )
    validate_phn_dataset = None
    if phn_embedding_config.add_phn:
        validate_phn_dataset = PhnDataset(name_manifest_path=validate_static_feature_dataset_config.name_manifest_path, phn_dir=phn_embedding_config.phn_dir)
    validate_pca_dataset = None
    if pca_config.learn_pca:
        validate_pca_dataset = NewFeatureDataset(name_manifest_path=validate_static_feature_dataset_config.name_manifest_path, feature_dir=pca_config.pca_label_dir, feature_name="pca_label")
    validate_emotion_dataset = None
    if emotion_config.add_emotion_embedding:
        validate_emotion_dataset = NewFeatureDataset(name_manifest_path=validate_static_feature_dataset_config.name_manifest_path, feature_dir=emotion_config.emotion_feature_dir, feature_name="emotion_index")
    
    validate_static_feature_dataset = CombinedFeatureDataset(validate_static_feature_dataset, validate_phn_dataset, validate_pca_dataset, validate_emotion_dataset, post_process_fn=check_feature_length_post_process_fn)
    validate_dataloader = DataLoader(dataset=validate_static_feature_dataset, batch_size=training_config.batch_size, shuffle=False, collate_fn=train_collate_fn, num_workers=training_config.num_workers)

    model1_pl = Model1PL(model1_config, training_config, train_static_feature_dataset_config)

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
        devices=training_config.devices,
        max_epochs=training_config.n_epochs,
        callbacks=callbacks,
        val_check_interval=training_config.val_check_interval,
        check_val_every_n_epoch=training_config.check_val_every_n_epoch,
        gradient_clip_val=training_config.gradient_clip_val,
        enable_checkpointing=training_config.checkpoint_config.save_checkpoint
    )

    trainer.fit(
        model=model1_pl,
        train_dataloaders=train_dataloader,
        val_dataloaders=validate_dataloader,
        ckpt_path=training_config.load_checkpoint_path if training_config.resume_training else None,
    )




if __name__ == "__main__":


    train()



































