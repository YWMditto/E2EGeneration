

import sys
sys.path.append(".")
import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path

from model_1 import (
    Model1Config,
    Model1
)

from data_prepare import (
    StaticFeatureDataset,
    StaticFeatureDatasetConfig,
    StaticFeatureCollater
)

from train_1 import (
    TrainingConfig, 
    UsedDatasetConfig,
    parse_config_from_yaml
)

from scripts.utils import _process_bar


def evaluate():
    """
    根据 static feature 生成对应的控制器标签然后存储；
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--pl_ckpt_path")
    parser.add_argument("--config_path")
    parser.add_argument("--evaluate_config")
    parser.add_argument("--save_dir")
    parser.add_argument("--device", type=eval, default=0)

    args = parser.parse_args()

    # 这里加载训练用的 config 只是为了加载模型；
    config_dict = parse_config_from_yaml(args.config_path, UsedDatasetConfig, Model1Config, TrainingConfig)
    # used_dataset_config: UsedDatasetConfig = config_dict["UsedDatasetConfig"]
    # validate_static_feature_dataset_config: StaticFeatureDatasetConfig = used_dataset_config.validate_dataset_config
    model1_config: Model1Config = config_dict['Model1Config']
    training_config: TrainingConfig = config_dict['TrainingConfig']

    pl_ckpt = torch.load(args.pl_ckpt_path)
    model1_state_dict = pl_ckpt["state_dict"]  # 这里如何操作取决于训练过程中如何保存模型参数，是直接保存整体的训练状态还是只保存模型参数；
    # 去除 prefix；
    prefix = "model1"
    len_prefix = len(prefix) + 1
    fixed_state_dict = {}
    for key, value in model1_state_dict.items():
        fixed_state_dict[key[len_prefix:]] = value
    model = Model1(model1_config, training_config)
    model.load_state_dict(fixed_state_dict)

    device = torch.device(f"cuda:{args.device}")
    model.to(device)
    model.eval()

    evaluate_config_dict = parse_config_from_yaml(args.evaluate_config, StaticFeatureDatasetConfig)
    evaluate_static_feature_dataset_config = evaluate_config_dict["StaticFeatureDatasetConfig"]
    evaluate_static_feature_dataset = StaticFeatureDataset(
        static_audio_feature_manifest_or_list=evaluate_static_feature_dataset_config.static_audio_feature_manifest_or_list,
        ctrl_manifest_or_list=evaluate_static_feature_dataset_config.ctrl_manifest_or_list,
        feature_rate=evaluate_static_feature_dataset_config.feature_rate,
        label_rate=evaluate_static_feature_dataset_config.label_rate,
        max_keep_feature_size=evaluate_static_feature_dataset_config.max_keep_feature_size,
        min_keep_feature_size=evaluate_static_feature_dataset_config.min_keep_feature_size
    )
    feature_path_list = evaluate_static_feature_dataset.audio_feature_path_list

    save_dir = Path(args.save_dir).joinpath(Path(args.pl_ckpt_path).stem)
    save_dir.mkdir(exist_ok=True, parents=True)

    lumi05_mouth_without_R_ctrl_indices = evaluate_static_feature_dataset_config.lumi05_mouth_without_R_ctrl_indices
    lumi05_eye_without_R_ctrl_indices = evaluate_static_feature_dataset_config.lumi05_eye_without_R_ctrl_indices

    lumi05_mouth_L_ctrl_indices = evaluate_static_feature_dataset_config.lumi05_mouth_L_ctrl_indices
    lumi05_eye_L_ctrl_indices = evaluate_static_feature_dataset_config.lumi05_eye_L_ctrl_indices
    lumi05_mouth_R_ctrl_indices = evaluate_static_feature_dataset_config.lumi05_mouth_R_ctrl_indices
    lumi05_eye_R_ctrl_indices = evaluate_static_feature_dataset_config.lumi05_eye_R_ctrl_indices
    
    with _process_bar("evaluate", total=len(evaluate_static_feature_dataset)) as update:

        with torch.no_grad():
            for idx, sample in enumerate(evaluate_static_feature_dataset):
                audio_feature = sample["audio_feature"]
                audio_feature = audio_feature.unsqueeze(0)
                padding_mask = torch.BoolTensor(size=audio_feature.shape[:2]).fill_(True)

                origin_ctrl_label = sample["ctrl_label"]
                ctrl_label = origin_ctrl_label.unsqueeze(0)
                mouth_ctrl_labels = ctrl_label[..., lumi05_mouth_without_R_ctrl_indices]
                eye_ctrl_labels = ctrl_label[..., lumi05_eye_without_R_ctrl_indices]
                
                collated_batch = {
                    "idx": torch.LongTensor([idx]).to(device),
                    "net_input": {
                        "audio_features": audio_feature.to(device),
                        "padding_mask": padding_mask.to(device),
                    },
                    "mouth_ctrl_labels": mouth_ctrl_labels.to(device),
                    "eye_ctrl_labels": eye_ctrl_labels.to(device)
                }

                output_dict = model.inference_step(collated_batch)
                mouth_ctrl_pred = output_dict["mouth_ctrl_pred"]
                eye_ctrl_pred = output_dict["eye_ctrl_pred"]
                mouth_ctrl_pred = mouth_ctrl_pred.cpu().squeeze(0)
                eye_ctrl_pred = eye_ctrl_pred.cpu().squeeze(0)

                origin_ctrl_label[..., lumi05_mouth_without_R_ctrl_indices] = mouth_ctrl_pred
                origin_ctrl_label[..., lumi05_eye_without_R_ctrl_indices] = eye_ctrl_pred

                origin_ctrl_label[..., lumi05_mouth_R_ctrl_indices] = origin_ctrl_label[..., lumi05_mouth_L_ctrl_indices]
                origin_ctrl_label[..., lumi05_eye_R_ctrl_indices] = origin_ctrl_label[..., lumi05_eye_L_ctrl_indices]

                torch.save(origin_ctrl_label, save_dir.joinpath(Path(feature_path_list[idx]).stem + ".pt"))

                update(1)



if __name__ == "__main__":
    evaluate()




















