

"""
TODO 实现对于 Model1 的训练，之后应当从中抽取出共用的 train_utils；
"""
import os
import sys
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

from torch.utils.data import DataLoader
import pytorch_lightning as pl


from data_prepare import (
    StaticFeatureDatasetConfig,
    StaticFeatureDataset,
    StaticFeatureCollater,
    ConstantTokenBatchSampler
)

from models import (
    Model1Config,
    Model1
)

from losses import LossConfig

from helper_fns import parse_config_from_yaml
from pl_modules import Model1PL



@dataclass
class TrainingConfig:

    use_constant_batch_sampler: bool = False
    one_batch_total_tokens: int = 6000

    batch_size: int = 8
    
    seed: int = 0
    shuffle: bool = True

    base_lr: float = 1e-4
    pre_proj_lr: float = 5e-5
    encoder_lr: float = 5e-5  # 用户在配置 encoder optim params 的时候应该使用 pipeline config；
    decoder_lr: float = 5e-5
    mouth_head_lr: float = 5e-5
    eye_head_lr: float = 5e-5
    weight_decay: float = 1e-5
    epoch_milestones: List[int] = field(default_factory=lambda: [1000])
    gamma: float = 0.5

    warmup_epochs: int = 8
    n_epochs: int = 2000

    evaluate_every: int = -4  # TODO
    gradient_accumulation_step: int = 1
    devices: List[int] = field(default_factory=lambda: [0])

    use_wandb: bool = False

    loss_config: LossConfig = LossConfig()


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
    model1_config: Model1Config = config_dict['Model1Config']
    training_config: TrainingConfig = config_dict['TrainingConfig']

    pl.seed_everything(training_config.seed)

    train_static_feature_dataset = StaticFeatureDataset(
        static_audio_feature_manifest_or_list=train_static_feature_dataset_config.static_audio_feature_manifest_or_list,
        ctrl_manifest_or_list=train_static_feature_dataset_config.ctrl_manifest_or_list,
        feature_rate=train_static_feature_dataset_config.feature_rate,
        label_rate=train_static_feature_dataset_config.label_rate,
        max_keep_feature_size=train_static_feature_dataset_config.max_keep_feature_size,
        min_keep_feature_size=train_static_feature_dataset_config.min_keep_feature_size
    )

    train_collate_fn = StaticFeatureCollater(
        max_feature_size=train_static_feature_dataset_config.max_feature_size,
        pad_feature=train_static_feature_dataset_config.pad_feature,
        random_crop=train_static_feature_dataset_config.random_crop
    )

    if training_config.use_constant_batch_sampler:
        train_batch_sampler = ConstantTokenBatchSampler(
            size_list=train_static_feature_dataset.size_list(),
            one_batch_total_tokens=training_config.one_batch_total_tokens,
            shuffle=training_config.shuffle,
            num_buckets=5,
            seed=training_config.seed,
            dataset_name="static feature dataset",
            num_replicas=len(training_config.devices),
            rank=int(os.environ.get("LOCAL_RANK", 0)),
            drop_last=True
        )
        train_dataloader = DataLoader(dataset=train_static_feature_dataset, batch_sampler=train_batch_sampler, collate_fn=train_collate_fn)
    else:
        train_dataloader = DataLoader(dataset=train_static_feature_dataset, batch_size=training_config.batch_size, shuffle=training_config.shuffle, collate_fn=train_collate_fn)

    model1_pl = Model1PL(model1_config, training_config, train_static_feature_dataset_config)
    

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=training_config.devices,
        max_epochs=training_config.n_epochs
    )

    trainer.fit(
        model=model1_pl,
        train_dataloaders=train_dataloader
    )




if __name__ == "__main__":


    train()



































