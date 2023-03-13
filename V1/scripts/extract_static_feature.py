


"""
从预训练模型中抽取固定的特征；


"""

import sys

from dataclasses import dataclass
from pathlib import Path
import soundfile as sf
from copy import deepcopy
from typing import List
import time

from torch.multiprocessing import Process, set_start_method, Queue, set_sharing_strategy


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    HubertConfig,
    HubertModel,
    Wav2Vec2Config,
    Wav2Vec2Model,
    WavLMConfig,
    WavLMModel,
    Wav2Vec2FeatureExtractor,
    AutoConfig,
    AutoModel
)

import logging
logger = logging.getLogger(__file__)


sys.path.append("V1/")
sys.path.append("..")
from data_prepare import AudioOnlyDataset, AudioWav2Vec2Collater, ConstantTokenBatchSampler, resample_feature_from_50_to_60
from utils import _process_bar, check_gpu_available, move_data_to_device, SearchFilePath


@dataclass
class ExtractFeatureConfig:
    pretrained_model_or_path: str
    save_root_dir: str
    one_batch_total_tokens: int = 2000000
    layer: int = 24


class TransformersPretrainedModel:

    def __init__(self, pretrained_model_or_path, device) -> None:
        
        self.model_config = AutoConfig.from_pretrained(pretrained_model_or_path)
        self.model = AutoModel.from_pretrained(pretrained_model_or_path)
        self.model.config.apply_spec_augment = False

        self.model.to(device)
        self.model.eval()
        self.device = device

    def forward(self, batch):
        audio_input = batch["audio_input"]
        outputs = self.model(**audio_input, output_hidden_states=True)
        return outputs


def make_save_name(file_path, dst_parent=None, join_parent=False, suffix=None):
    file_path = Path(file_path)

    if dst_parent is None:
        name = file_path.stem
        if suffix is not None:
            name += suffix
        return name

    parent_path_list = [file_path.stem]
    while str(file_path) != '/':
        if file_path.parent.stem != dst_parent:
            file_path = file_path.parent 
            parent_path_list.append(file_path.stem)
        else:
            if join_parent:
                parent_path_list.append(file_path.parent.stem)
            break
    
    parent_path_list = parent_path_list[::-1]
    name = "<|>".join(parent_path_list)
    if suffix is not None:
        name += suffix
    return name



def extract_feature(audio_manifest_or_list, extract_feature_config: ExtractFeatureConfig, device: int, num_replicas: int, rank: int):
    
    save_path = Path(extract_feature_config.save_root_dir)
    layer = extract_feature_config.layer

    device = torch.device(f"cuda:{device}")

    model = TransformersPretrainedModel(extract_feature_config.pretrained_model_or_path, device)

    dataset = AudioOnlyDataset(audio_manifest_or_list, sample_rate=16000, normalize=False, raw_audio=True)
    audio_path_list = dataset.audio_path_list

    batch_sampler = ConstantTokenBatchSampler(
        size_list=dataset.size_list(), 
        one_batch_total_tokens=extract_feature_config.one_batch_total_tokens,
        num_replicas=num_replicas,
        rank=rank,
        drop_last=False
    )
    audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(extract_feature_config.pretrained_model_or_path)
    collate_fn = AudioWav2Vec2Collater(audio_feature_extractor)
    dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    with _process_bar(task_name="extract feature", total=len(dataloader), pid=rank) as update:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch = move_data_to_device(batch, device=device, dtype=torch.Tensor)

                outputs = model.forward(batch)
                hidden_states = outputs.hidden_states[layer]
                # hidden_states = outputs.last_hidden_state
                frame_lengths = batch["frame_lengths"]
                features = [cur_h[:jj].cpu() for cur_h in hidden_states for jj in frame_lengths]

                # save feature
                idx_list = batch["idx"]
                audio_paths = [audio_path_list[w] for w in idx_list]
                save_names = [make_save_name(w, suffix=".pt") for w in audio_paths]
                for j in range(len(save_names)):
                    torch.save(features[j], save_path.joinpath(save_names[j]))

                update(1)


class ExtractFeatureProcess(Process):
    def __init__(self, audio_manifest_or_list, extract_feature_config, device: int, num_replicas: int, rank: int, **kwargs) -> None:
        super(ExtractFeatureProcess, self).__init__(**kwargs)

        logger.info(f"rank / num_replicas: {rank} / {num_replicas}")

        self.audio_manifest_or_list = audio_manifest_or_list
        self.extract_feature_config = extract_feature_config
        self.device = device
        self.num_replicas = num_replicas
        self.rank = rank

    def run(self) -> None:
        extract_feature(
            audio_manifest_or_list=self.audio_manifest_or_list,
            extract_feature_config=self.extract_feature_config,
            device=self.device,
            num_replicas=self.num_replicas,
            rank=self.rank
        )


def parallel_process(audio_manifest_or_list, extract_feature_config, devices: List[int], num_replicas: int, tasks=None):
    save_path = Path(extract_feature_config.save_root_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    tasks = list(range(num_replicas)) if tasks is None else tasks

    a_devices = deepcopy(devices)
    a_devices.reverse()

    running_processes = []

    while len(tasks):
        if len(a_devices):
            cur_device = a_devices.pop()
            if check_gpu_available(cur_device, 20*1024**3):
                cur_task = tasks.pop(0)
                p = ExtractFeatureProcess(
                    audio_manifest_or_list=audio_manifest_or_list,
                    extract_feature_config=extract_feature_config,
                    device=cur_device,
                    num_replicas=num_replicas,
                    rank=cur_task
                )
                p.start()
                running_processes.append(p)
        else:
            time.sleep(60)

            a_devices = deepcopy(devices)
            a_devices.reverse() 
  
    for p in running_processes:
        p.join()



def hubert_extract(pretrained_model_or_path, layer, save_root_dir, devices):
    audio_manifest_or_list = "/data/lipsync/xgyang/e2e_data/static_feature/aligned_audios_manifest.txt"
    extract_feature_config = ExtractFeatureConfig(
        pretrained_model_or_path=pretrained_model_or_path,
        one_batch_total_tokens=2000000,
        save_root_dir=save_root_dir,
        layer=layer
    )
    parallel_process(
        audio_manifest_or_list=audio_manifest_or_list,
        extract_feature_config=extract_feature_config,
        devices=devices,
        num_replicas=len(devices),
        tasks=None
    )

def wavlm_extract(layer, save_root_dir, devices):
    audio_manifest_or_list = "/data/lipsync/xgyang/e2e_data/static_feature/aligned_audios_manifest.txt"
    extract_feature_config = ExtractFeatureConfig(
        pretrained_model_or_path="/data/lipsync/xgyang/WavLM/wavlm_pretrained_ckpt/wavlm_cn_large_ckpt_path",
        one_batch_total_tokens=2000000,
        save_root_dir=save_root_dir,
        layer=layer
    )
    parallel_process(
        audio_manifest_or_list=audio_manifest_or_list,
        extract_feature_config=extract_feature_config,
        devices=devices,
        num_replicas=len(devices),
        tasks=None
    )

def wav2vec2_extract(layer, save_root_dir, devices):
    audio_manifest_or_list = "/data/lipsync/xgyang/e2e_data/static_feature/aligned_audios_manifest.txt"
    extract_feature_config = ExtractFeatureConfig(
        pretrained_model_or_path="TencentGameMate/chinese-wav2vec2-large",
        one_batch_total_tokens=2000000,
        save_root_dir=save_root_dir,
        layer=layer
    )
    parallel_process(
        audio_manifest_or_list=audio_manifest_or_list,
        extract_feature_config=extract_feature_config,
        devices=devices,
        num_replicas=len(devices),
        tasks=None
    )


def resample_feature(feature_manifest_path, manifest_save_path, new_feature_save_dir):
    """
    TODO 这里先直接使用 resample_feature_from_50_to_60；
    """

    manifest_save_path = Path(manifest_save_path)
    manifest_save_path.parent.mkdir(exist_ok=True, parents=True)
    new_feature_save_dir = Path(new_feature_save_dir)
    new_feature_save_dir.mkdir(exist_ok=True, parents=True)

    manifest_f = open(manifest_save_path, "w")
    with open(feature_manifest_path, "r") as f:
        for line in f:
            feature_path, feature_size = line.rstrip().split("\t")
            feature = torch.load(feature_path)
            resampled_feature = resample_feature_from_50_to_60(feature)
            feature_path = Path(feature_path)
            new_save_name = new_feature_save_dir.joinpath(Path(feature_path).name)
            torch.save(resampled_feature, new_save_name)
            new_line = str(new_save_name) + "\t" + str(len(resampled_feature))
            print(new_line, file=manifest_f)
    manifest_f.close()


# def extract_lumi05_real_ctrl_indices(ctrl_path_dirs, map_save_path, new_ctrl_save_dir):
#     from data_prepare import StaticFeatureConfig

#     if isinstance(ctrl_path_dirs, (str, Path)):
#         ctrl_path_dirs = [ctrl_path_dirs]

#     search_model = SearchFilePath()
#     ctrl_path_list = search_model.find_all_file_paths(
#         data_dirs=ctrl_path_dirs,
#         pattern=".+\.pt",
#         n_proc=64,
#         depth=1
#     )

#     new_ctrl_save_dir = Path(new_ctrl_save_dir)
#     new_ctrl_save_dir.mkdir(exist_ok=True, parents=True)

#     Path(map_save_path).mkdir(exist_ok=True, parents=True)
#     map_save_f = open(map_save_path, 'w')

#     mouth_indices = StaticFeatureConfig.lumi05_mouth_ctrl_indices
#     eye_indices = StaticFeatureConfig.lumi05_eye_ctrl_indices
#     with _process_bar("extract lumi05 real ctrl indices", total=len(ctrl_path_list)) as update:
#         for path in ctrl_path_list:
#             ctrl_label = torch.load(path)
#             cur_mouth_ctrl = ctrl_label[:, mouth_indices]
#             cur_eye_ctrl = ctrl_label[:, eye_indices]




def _extract_state_dict_from_ser_hubert(ckpt_dir, save_dir):

    ckpt_dir = Path(ckpt_dir)

    model_state_dict = torch.load(ckpt_dir.joinpath("pytorch_model.bin"))


    hubert_model = HubertModel.from_pretrained(ckpt_dir)

    a = 1










if __name__ == "__main__":

    # set_start_method("spawn", force=True)
    # set_sharing_strategy('file_system')

    hubert_extract(
        "/data/lipsync/xgyang/E2EGeneration/ser_dir/hubert-large-CN-14-emotions-corrected-1028/checkpoint-17700",
        24, 
        "/data/lipsync/xgyang/e2e_data/static_feature/ser_hubert/50/layer24", 
        [3,4,5,6,7]
    )
    # wavlm_extract(6, "/data/lipsync/xgyang/e2e_data/static_feature/layer6/wavlm_50", [5,6,7])
    # wav2vec2_extract(6, "/data/lipsync/xgyang/e2e_data/static_feature/layer6/wav2vec2_50", [5,6,7])

    # resample_feature(
    #     feature_manifest_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_hubert_50.txt",
    #     manifest_save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_hubert_60.txt",
    #     new_feature_save_dir="/data/lipsync/xgyang/e2e_data/static_feature/hubert_60"
    # )

    # resample_feature(
    #     feature_manifest_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_wavlm_50.txt",
    #     manifest_save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_wavlm_60.txt",
    #     new_feature_save_dir="/data/lipsync/xgyang/e2e_data/static_feature/wavlm_60"
    # )

    # resample_feature(
    #     feature_manifest_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_wav2vec2_50.txt",
    #     manifest_save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_wav2vec2_60.txt",
    #     new_feature_save_dir="/data/lipsync/xgyang/e2e_data/static_feature/wav2vec2_60"
    # )



    # _extract_state_dict_from_ser_hubert(
    #     ckpt_dir="/data/lipsync/xgyang/E2EGeneration/ser_dir/hubert-large-CN-14-emotions-corrected-1028/checkpoint-17700",
    #     save_dir="/data/lipsync/xgyang/E2EGeneration/ser_dir/extracted_direct_hubert/hubert_large_1"
    # )






















