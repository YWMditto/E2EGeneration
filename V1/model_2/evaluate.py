

import sys
sys.path.extend([".", ".."])
import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path

from model_2 import (
    Model2Config,
    Model2
)

from data_prepare import (
    RawAudioDataset,
    RawAudioDatasetConfig,
    RawAudioCollater,

    norm_decode,
)

from train import (
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
    parser.add_argument("--train_config_path")
    parser.add_argument("--eval_name_manifest_path")
    parser.add_argument("--eval_audio_dir")
    parser.add_argument("--eval_ctrl_label_dir")
    parser.add_argument("--eval_normalize")
    parser.add_argument("--lumi_template_path")
    parser.add_argument("--save_dir")
    parser.add_argument("--device", type=eval, default=0)

    args = parser.parse_args()

    # 这里加载训练用的 config 只是为了加载模型；
    config_dict = parse_config_from_yaml(args.train_config_path, UsedDatasetConfig, Model2Config, TrainingConfig)
    # used_dataset_config: UsedDatasetConfig = config_dict["UsedDatasetConfig"]
    # validate_raw_audio_dataset_config: StaticFeatureDatasetConfig = used_dataset_config.validate_dataset_config
    model1_config: Model2Config = config_dict['Model2Config']
    training_config: TrainingConfig = config_dict['TrainingConfig']

    pl_ckpt = torch.load(args.pl_ckpt_path)
    model1_state_dict = pl_ckpt["state_dict"]  # 这里如何操作取决于训练过程中如何保存模型参数，是直接保存整体的训练状态还是只保存模型参数；
    # 去除 prefix；
    prefix = "model2"
    len_prefix = len(prefix) + 1
    fixed_state_dict = {}
    for key, value in model1_state_dict.items():
        fixed_state_dict[key[len_prefix:]] = value
    model = Model2(model1_config, training_config)
    model.load_state_dict(fixed_state_dict)

    device = torch.device(f"cuda:{args.device}")
    model.to(device)
    model.eval()

    evaluate_raw_audio_dataset_config = RawAudioDatasetConfig(
        name_manifest_path=args.eval_name_manifest_path,
        audio_dir=args.eval_audio_dir,
        ctrl_label_dir=args.eval_ctrl_label_dir,
        normalize=args.eval_normalize
    )
    evaluate_raw_audio_dataset = RawAudioDataset(
        name_manifest_path=evaluate_raw_audio_dataset_config.name_manifest_path,
        audio_dir=evaluate_raw_audio_dataset_config.audio_dir,
        ctrl_label_dir=evaluate_raw_audio_dataset_config.ctrl_label_dir,
    )

    names_list = evaluate_raw_audio_dataset.names

    save_dir = Path(args.save_dir).joinpath(Path(args.pl_ckpt_path).stem)
    save_dir.mkdir(exist_ok=True, parents=True)

    lumi_template = torch.load(args.lumi_template_path)
    lumi_template = lumi_template.unsqueeze(0)

    lumi05_mouth_without_R_ctrl_indices = evaluate_raw_audio_dataset_config.lumi05_mouth_without_R_ctrl_indices
    lumi05_eye_without_R_ctrl_indices = evaluate_raw_audio_dataset_config.lumi05_eye_without_R_ctrl_indices

    lumi05_mouth_L_ctrl_indices = evaluate_raw_audio_dataset_config.lumi05_mouth_L_ctrl_indices
    lumi05_eye_L_ctrl_indices = evaluate_raw_audio_dataset_config.lumi05_eye_L_ctrl_indices
    lumi05_mouth_R_ctrl_indices = evaluate_raw_audio_dataset_config.lumi05_mouth_R_ctrl_indices
    lumi05_eye_R_ctrl_indices = evaluate_raw_audio_dataset_config.lumi05_eye_R_ctrl_indices

    ori_min = torch.FloatTensor(evaluate_raw_audio_dataset_config.norm_ori_min)
    tgt_min = torch.FloatTensor([evaluate_raw_audio_dataset_config.norm_tgt_min])
    remap_scale = torch.FloatTensor(evaluate_raw_audio_dataset_config.norm_remap_scale)
    
    with _process_bar("evaluate", total=len(evaluate_raw_audio_dataset)) as update:

        with torch.no_grad():
            for idx, sample in enumerate(evaluate_raw_audio_dataset):
                audio = sample["audio"]
                audio = audio.unsqueeze(0)
                padding_mask = torch.BoolTensor(size=audio.shape).fill_(True)

                collated_batch = {
                    "idx": torch.LongTensor([idx]).to(device),
                    "net_input": {
                        "audios": audio.to(device),
                        "padding_mask": padding_mask.to(device),
                    },
                }

                output_dict = model.inference_step(collated_batch)
                mouth_ctrl_pred = output_dict["mouth_ctrl_pred"]
                mouth_ctrl_pred = mouth_ctrl_pred.cpu().squeeze(0)

                if training_config.learn_eye:
                    eye_ctrl_pred = output_dict["eye_ctrl_pred"]
                    eye_ctrl_pred = eye_ctrl_pred.cpu().squeeze(0)
                else:
                    eye_ctrl_pred = torch.zeros((len(mouth_ctrl_pred), len(lumi05_eye_without_R_ctrl_indices)))

                mix_pred = torch.cat([mouth_ctrl_pred, eye_ctrl_pred], dim=-1)
                mix_pred = norm_decode(mix_pred, ori_min=ori_min, tgt_min=tgt_min, remap_scale=remap_scale)
                mouth_ctrl_pred = mix_pred[..., :len(lumi05_mouth_without_R_ctrl_indices)]
                eye_ctrl_pred = mix_pred[..., len(lumi05_mouth_without_R_ctrl_indices):]

                origin_ctrl_label = lumi_template.clone()
                origin_ctrl_label = origin_ctrl_label.repeat((len(mouth_ctrl_pred), 1))
                
                origin_ctrl_label[..., lumi05_mouth_without_R_ctrl_indices] = mouth_ctrl_pred
                origin_ctrl_label[..., lumi05_mouth_R_ctrl_indices] = origin_ctrl_label[..., lumi05_mouth_L_ctrl_indices]

                if training_config.learn_eye:
                    origin_ctrl_label[..., lumi05_eye_without_R_ctrl_indices] = eye_ctrl_pred
                    origin_ctrl_label[..., lumi05_eye_R_ctrl_indices] = origin_ctrl_label[..., lumi05_eye_L_ctrl_indices]

                torch.save(origin_ctrl_label, save_dir.joinpath(names_list[idx] + ".pt"))

                update(1)



if __name__ == "__main__":
    evaluate()




















