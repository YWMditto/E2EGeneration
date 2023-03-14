

"""

使用 normalized extracted ctrl labels 当做输入；

输出是 /data/lipsync/xgyang/e2e_data/emotions，其来源为 /data/lipsync/xgyang/E2EGeneration/V1/scripts/prepare_emotion_embedding.py，是我们根据文件名预先处理好的情绪特征；

"""


import sys
sys.path.append("..")

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from data_prepare import load_data_directly


@dataclass
class EmotionDatasetConfig:

    name_manifest_path: Optional[str] = None
    ctrl_label_dir: Optional[str] = None
    emotion_label_dir: Optional[str] = None



class EmotionDataset(Dataset):

    def __init__(
        self,
        name_manifest_path: Optional[str] = None,
        ctrl_label_dir: Optional[str] = None,
        emotion_label_dir: Optional[str] = None,
        ctrl_label_suffix: str = ".pt",
        emotion_label_suffix: str = ".pt"
    ) -> None:

        self.names = load_data_directly(name_manifest_path)
        self.ctrl_label_dir = Path(ctrl_label_dir)
        self.emotion_label_dir = Path(emotion_label_dir)

        self.ctrl_label_suffix = ctrl_label_suffix
        self.emotion_label_suffix = emotion_label_suffix
    
    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        ctrl_label = torch.load(self.ctrl_label_dir.joinpath(name + self.ctrl_label_suffix))
        emotion_label = torch.load(self.emotion_label_dir.joinpath(name + self.emotion_label_suffix))
        return {"idx": index, "ctrl_label": ctrl_label, "emotion_label": emotion_label}
    


class EmotionCollater:
    def __init__(
        self,
    ) -> None:
        
        ...

    def __call__(self, batch):

        # ctrl label 是一个时间序列，因此需要在时间上进行 pad；

        ctrl_label_list = [s["ctrl_label"] for s in batch]
        ctrl_label_length_list = [len(s) for s in ctrl_label_list]
        ctrl_label_max_length = max(ctrl_label_length_list)
        collated_ctrl_labels = torch.zeros(size=(len(ctrl_label_list), ctrl_label_max_length, ctrl_label_list[0].size(1)))

        for i in range(len(ctrl_label_list)):
            cur_ctrl_label = ctrl_label_list[i]
            cur_length = len(cur_ctrl_label)

            if cur_length == ctrl_label_max_length:
                collated_ctrl_labels[i] = cur_ctrl_label
            else:
                collated_ctrl_labels[i, :cur_length] = cur_ctrl_label

        collated_batch = {
            "idx": [s["idx"] for s in batch],
            "collated_ctrl_labels": collated_ctrl_labels,
            "ctrl_label_lengths": torch.LongTensor(ctrl_label_length_list),
            "emotion_labels": torch.LongTensor([s["emotion_label"] for s in batch])
        }

        return collated_batch
        
        












