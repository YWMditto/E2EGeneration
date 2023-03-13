
import sys
import logging
from typing import Optional, List, Union, Callable
from pathlib import Path
import soundfile as sf
from itertools import chain
import numpy as np
from math import ceil
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


logger = logging.getLogger(__file__)


 # 在 2007 控制器向量中对应的坐标；
LUMI05_MOUTH_WITHOUT_R_CTRL_INDICES = [
    513, 514, 522, 532, 541, 559, 567, 568, 577, 586, 613, 622, 631, 640, 649, 666, 667, 676, 685, 730, 739, 
    748, 757, 766, 775, 784, 802, 811, 820, 829, 838, 847, 874, 883, 892, 901, 910, 919, 928, 937, 982, 1000, 
    1027, 1036, 1045, 1054, 1063, 1072, 1081, 1089, 1090, 1099,
]
LUMI05_EYE_WITHOUT_R_CTRL_INDICES = [
    1, 9, 18, 27, 37, 46, 55, 145, 154, 163, 190, 199, 486, 487,
]
LUMI05_MOUTH_L_CTRL_INDICES = [
    613, 622, 631, 640, 649, 666, 667, 676, 685, 730, 739, 
    748, 757, 766, 775, 784, 802, 811, 820, 829, 838, 847, 874, 883, 892, 901, 910, 919, 928, 937, 982, 1000, 
    1027, 1036, 1045, 1054, 1063, 1072, 1081, 1089, 1090, 1099,
]
LUMI05_EYE_L_CTRL_INDICES = [
    1, 9, 18, 27, 37, 46, 55, 145, 154, 163, 190, 199
]
LUMI05_MOUTH_R_CTRL_INDICES = [
    1108, 1117, 1126, 1135, 1144, 1161, 1162, 1171, 1180, 1225, 1234, 1243, 1252, 1261, 1270, 1279, 1297, 1306, 
    1315, 1324, 1333, 1342, 1369, 1378, 1387, 1396, 1405, 1414, 1423, 1432, 1477, 1495, 1522, 1531, 1540, 1549, 
    1558, 1567, 1576, 1584, 1585, 1594,
]
LUMI05_EYE_R_CTRL_INDICES = [
    73, 81, 90, 99, 109, 118, 127, 208, 217, 226, 253, 262,
]
NORM_ORI_MAX = [0.0533, 0.7664, 0.7560, 0.1297, 0.0942, 0.1288, 0.2982, 0.6165, 0.3286,
    0.4970, 0.4706, 0.2251, 0.8300, 0.4386, 0.8027, 0.1630, 0.5142, 0.3098,                                                                                                                                                
    1.0875, 0.1654, 0.2118, 0.6030, 0.9323, 0.9986, 0.6012, 0.4645, 0.4441,
    0.4699, 0.6974, 0.3942, 1.2352, 1.2274, 1.0911, 0.2452, 0.2529, 0.5024,
    0.4905, 0.6924, 0.6739, 0.6496, 0.5537, 0.2048, 0.3744, 0.1697, 0.9325,
    0.8312, 1.2532, 1.2055, 0.8853, 0.3692, 0.6112, 0.8275, 0.4963, 1.6168,
    2.1383, 1.2824, 0.7338, 0.7811, 0.5666, 0.7617, 0.9788, 0.8382, 0.2267,
    1.2226, 0.8984, 0.7317]
NORM_ORI_MIN = [-0.0333, -0.0269, -0.1450, -0.2116, -0.0057, -0.0115, -0.0888, -1.0891,
    -0.3678, -0.2380, -0.0720, -0.3273, -0.1502, -0.0436, -0.0594, -0.3844,
    -0.1418, -0.0469, -0.0909, -0.9758, -0.7919, -0.1354, -0.1597, -0.1417,
    -0.1516, -0.1061, -0.1646, -0.0722, -0.2244, -0.5679, -0.2677, -0.2728,
    -0.2733, -0.1495, -0.1590, -0.0557, -0.0457, -0.3390, -0.4213, -0.1025,
    -0.0558, -0.7074, -0.1653, -0.4797, -0.1342, -0.1918, -0.1677, -0.1858,
    -0.0883, -0.1052, -0.1572, -0.2134, -0.0244,  0.6717,  0.8378,  0.9538,
    -0.0184, -0.0424, -0.0772, -1.0264, -0.1486, -0.2171, -0.3234, -0.1251,
    -0.8869, -0.8562]
NORM_TGT_MAX: float = 1.
NORM_TGT_MIN: float = -1.
NORM_MAP_SCALE = [23.0920,  2.5212,  2.2197,  5.8597, 20.0263, 14.2639,  5.1676,  1.1726,
        2.8716,  2.7210,  3.6860,  3.6202,  2.0405,  4.1471,  2.3198,  3.6539,
        3.0490,  5.6076,  1.6972,  1.7526,  1.9927,  2.7088,  1.8315,  1.7539,
        2.6570,  3.5052,  3.2856,  3.6892,  2.1697,  2.0789,  1.3308,  1.3332,
        1.4659,  5.0683,  4.8565,  3.5834,  3.7298,  1.9391,  1.8261,  2.6592,
        3.2810,  2.1925,  3.7056,  3.0795,  1.8750,  1.9551,  1.4075,  1.4376,
        2.0542,  4.2156,  2.6030,  1.9215,  3.8413,  2.1163,  1.5379,  6.0866,
        2.6592,  2.4286,  3.1066,  1.1185,  1.7740,  1.8953,  3.6357,  1.4840,
        1.1202,  1.2595]
NORM_REMAP_SCALE = [0.0433, 0.3966, 0.4505, 0.1707, 0.0499, 0.0701, 0.1935, 0.8528, 0.3482,
    0.3675, 0.2713, 0.2762, 0.4901, 0.2411, 0.4311, 0.2737, 0.3280, 0.1783,
    0.5892, 0.5706, 0.5018, 0.3692, 0.5460, 0.5702, 0.3764, 0.2853, 0.3044,
    0.2711, 0.4609, 0.4810, 0.7514, 0.7501, 0.6822, 0.1973, 0.2059, 0.2791,
    0.2681, 0.5157, 0.5476, 0.3761, 0.3048, 0.4561, 0.2699, 0.3247, 0.5333,
    0.5115, 0.7105, 0.6956, 0.4868, 0.2372, 0.3842, 0.5204, 0.2603, 0.4725,
    0.6502, 0.1643, 0.3761, 0.4118, 0.3219, 0.8941, 0.5637, 0.5276, 0.2751,
    0.6739, 0.8927, 0.7940]



def load_sequence_data(manifest_or_list, max_keep_sample_size=None, min_keep_sample_size=None, load_length: bool = True):
    if max_keep_sample_size is not None and min_keep_sample_size is not None:
        assert 0 <= min_keep_sample_size < max_keep_sample_size

    path_list = []
    size_list = [] if load_length else None

    longer_num = 0
    shorter_num = 0
    with open(manifest_or_list, 'r') as f:
        for line in f:
            line = line.rstrip().split('\t')
            if load_length:
                _path, size = line
                size = int(size)
                if max_keep_sample_size is not None and size > max_keep_sample_size:
                    longer_num += 1
                elif min_keep_sample_size is not None and size < min_keep_sample_size:
                    shorter_num += 1
                else:
                    path_list.append(str(_path))
                    size_list.append(size)
            else:
                path_list.append(str(line[0]))
            
    
    if load_length:
        logger.info(
            (
                f"max_keep={max_keep_sample_size}, min_keep={min_keep_sample_size}, "
                f"loaded {len(path_list)}, skipped {shorter_num} short and {longer_num} long, "
                f"longest-loaded={max(size_list)}, shortest-loaded={min(size_list)}"
            )
        )
    else:
        logger.info(
            (
                f"loaded {len(path_list)}."
            )
        )

    return path_list, size_list


def verify_label_lengths(audio_size_list, audio_path_list, ctrl_size_list, ctrl_path_list, sample_rate, label_rate, tol=0.1):
    assert sample_rate > 0 and label_rate > 0

    assert len(audio_size_list) == len(audio_path_list) == len(ctrl_size_list) == len(ctrl_path_list), \
        f"序列长度不一致: audio size/path, ctrl size/path: {len(audio_size_list)}/{len(audio_path_list)}, {ctrl_size_list}/{ctrl_path_list}"

    num_invalid = 0
    for i in range(len(audio_size_list)):
        if abs((audio_seconds := audio_size_list[i] / sample_rate) - (ctrl_seconds := ctrl_size_list[i] / label_rate)) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{audio_seconds} - {ctrl_seconds}| > {tol}) "
                    f"of audio {audio_path_list[i]} and ctrl {ctrl_path_list[i]}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_size_list[i]}; "
                    f"label length = {ctrl_size_list[i]}."
                )
            )

            num_invalid += 1
    
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )



@dataclass
class RawAudioDatasetConfig:
    # 为了能够直接初始化这些 conifg，所有的值如果确实没有默认值全部设置成 None；
    audio_manifest_or_list: Optional[str] = None
    ctrl_manifest_or_list: Optional[str] = None
    sample_rate: Optional[int] = None
    label_rate: Optional[int] = None
    max_keep_sample_size: Optional[int] = None
    min_keep_sample_size: Optional[int] = None
    normalize: bool = False
    
    # 在 2007 控制器向量中对应的坐标；
    lumi05_mouth_without_R_ctrl_indices: List[int] = field(default_factory=lambda: LUMI05_MOUTH_WITHOUT_R_CTRL_INDICES)
    lumi05_eye_without_R_ctrl_indices: List[int] = field(default_factory=lambda: LUMI05_EYE_WITHOUT_R_CTRL_INDICES)
    
    lumi05_mouth_L_ctrl_indices: List[int] = field(default_factory=lambda: LUMI05_MOUTH_L_CTRL_INDICES)
    lumi05_eye_L_ctrl_indices: List[int] = field(default_factory=lambda: LUMI05_EYE_L_CTRL_INDICES)
    
    lumi05_mouth_R_ctrl_indices: List[int] = field(default_factory=lambda: LUMI05_MOUTH_R_CTRL_INDICES)
    lumi05_eye_R_ctrl_indices: List[int] = field(default_factory=lambda: LUMI05_EYE_R_CTRL_INDICES)

    norm_ori_max: List[float] = field(default_factory=lambda: NORM_ORI_MAX)
    norm_ori_min: List[float] = field(default_factory=lambda: NORM_ORI_MIN)
    norm_tgt_max: float = 1.
    norm_tgt_min: float = -1.
    norm_map_scale: List[float] = field(default_factory=lambda: NORM_MAP_SCALE)
    norm_remap_scale: List[float] = field(default_factory=lambda: NORM_REMAP_SCALE)

    # collate config
    max_feature_size: Optional[int] = None
    pad_feature: bool = True
    random_crop: bool = False
    



class RawAudioDataset(Dataset):

    """
    加载训练数据，音频和实际的控制器数据都是在实际使用时才会读取；
    
    可以指定是加载原始音频还是加载从预训练模型中抽取出来的音频特征；
        如果是加载固定的音频特征，那么注意此时的 sample_rate 表示的实际上是 label_rate 的意思，即表示 1 s内多少音频特征；


    Note: 控制器总体的数量很多（2007），但是在实际训练的时候我们只需要其中的一部分，例如 eye 和 mouth，这里数据集不需要管这些操作，如果需要在每次拿到数据的时候再抽取，
     那么需要在实际训练的时候进行额外的处理；

    因为一些数据的控制器与实际的音频无法对齐，因此需要进行切割，这里放在实际的模型中；
    注意此时因为我们没办法提前拿到 feature，因此实际的从 50 hz 采样到 60 hz 需要在模型内部完成；
            
    """

    def __init__(
        self,
        audio_manifest_or_list: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        ctrl_manifest_or_list: Union[str, Path, List[Union[str, Path]]] = None,
        sample_rate: Optional[int] = 16000,
        label_rate: Optional[int] = 60, 
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None, 
        normalize: bool = False, 
    ):
        
        # if audio_manifest_or_list is None and static_audio_feature_manifest_or_list is None:
        #     error_meg = f"audio_manifest_or_list and static_audio_feature_manifest_or_list are all None."
        #     logger.error(error_meg)
        #     raise ValueError(error_meg)

        self.audio_path_list, self.audio_size_list = load_sequence_data(manifest_or_list=audio_manifest_or_list, 
                                                                max_keep_sample_size=max_keep_sample_size, min_keep_sample_size=min_keep_sample_size)
        self.ctrl_path_list, self.ctrl_size_list = load_sequence_data(manifest_or_list=ctrl_manifest_or_list)

        if sample_rate is not None and label_rate is not None:
            verify_label_lengths(audio_size_list=self.audio_size_list, audio_path_list=self.audio_path_list, ctrl_size_list=self.ctrl_size_list, 
                                ctrl_path_list=self.ctrl_path_list, sample_rate=sample_rate, label_rate=label_rate, tol=0.1)
        
        self.audio_manifest_or_list = audio_manifest_or_list
        self.ctrl_manifest_or_list = ctrl_manifest_or_list
        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.max_keep_sample_size = max_keep_sample_size
        self.min_keep_sample_size = min_keep_sample_size
        self.normalize = normalize

    def get_audio(self, index):
        audio_path = self.audio_path_list[index]
        audio, cur_sample_rate = sf.read(audio_path)
        audio = torch.from_numpy(audio).float()
        audio = self.postprocess(audio, cur_sample_rate)
        return audio

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

    def get_ctrl_label(self, index):
        return torch.load(self.ctrl_path_list[index])
    
    def __len__(self):
        return len(self.audio_path_list)
    
    def __getitem__(self, index):
        audio = self.get_audio(index)
        ctrl_label = self.get_ctrl_label(index)
        return {"idx": index, "audio": audio, "ctrl_label": ctrl_label}

    def num_tokens(self, index) -> int:
        # 用来 order indices；
        return self.audio_size_list[index]

    def size_list(self):
        # 用来 order indices，collate；
        return self.audio_size_list




def resample_feature_from_50_to_60(feature: torch.Tensor):
    """
    对于长为 length 的 feature，先将 0 ~ length - 1 映射到 60 hz 的坐标上，查看哪些向量可以直接赋值；
    然后剩下的不能赋值的向量，根据其周围四个向量进行插值得到：
        1：0 0 2 3
        7：5 6 8 9 (5 6 8 8)

    feature: [N, H]
    return: [ceil((N-1)*1.2) + 1, H]
    """
    assert len(feature.shape) == 2

    length = len(feature)
    directly_map_indices = [ceil( w *1.2) for w in range(length)]
    interpolate_indices = []
    idx = 0

    mapped_length = ceil((length - 1) * 1.2)
    while (int_idx := 1 + idx *6) < mapped_length:
        interpolate_indices.append(int_idx)
        idx += 1

    computed_indices = []
    for int_idx in interpolate_indices:
        computed_indices.append([max(int_idx -2, 0), max(int_idx -1, 0), min(int_idx +1, mapped_length), min(int_idx +2, mapped_length)])
    computed_indices = torch.LongTensor(computed_indices).to(feature.device)

    interpolated_feature = feature.new_zeros((mapped_length +1, feature.size(1)))
    interpolated_feature[directly_map_indices] = feature

    weights = torch.FloatTensor([1 /6, 1/ 3, 1 / 3, 1 / 6]).unsqueeze(0).unsqueeze(2)
    computed_feature_value = (interpolated_feature[computed_indices] * weights).sum(dim=1)
    interpolated_feature[interpolate_indices] = computed_feature_value

    return interpolated_feature



def norm_encode(data: Union[torch.Tensor, np.ndarray], ori_max=None, ori_min=None, tgt_max=None, tgt_min=None, map_scale=None):
    """
    根据统计量将原始特征进行 normalize；

    data: [N, H], [H]

    x_tgt = (x_ori - ori_min) / (ori_max - ori_min) * (tgt_max - tgt_min) + tgt_min
    """
    if map_scale is not None:
        return (data - ori_min) * map_scale + tgt_min
    else:
        return (data - ori_min) * ((tgt_max - tgt_min) / (ori_max - ori_min))  + tgt_min



def norm_decode(data: Union[torch.Tensor, np.ndarray], ori_max=None, ori_min=None, tgt_max=None, tgt_min=None, remap_scale=None):

    """
    根据统计量将 normalize 后的值重新映射会原始的值；

    x_ori = (x_tgt - tgt_min) / (tgt_max - tgt_min) * (ori_max - ori_min) + ori_min
    """
    if remap_scale is not None:
        return (data - tgt_min) * remap_scale + ori_min
    else:
        return (data - tgt_min) * ((ori_max - ori_min) / (tgt_max - tgt_min)) + ori_min




@dataclass
class StaticFeatureDatasetConfig:
    # 为了能够直接初始化这些 conifg，所有的值如果确实没有默认值全部设置成 None；
    static_audio_feature_manifest_or_list: Optional[str] = None
    ctrl_manifest_or_list: Optional[str] = None
    feature_rate: Optional[int] = None
    label_rate: Optional[int] = None
    max_keep_feature_size: Optional[int] = None
    min_keep_feature_size: Optional[int] = None
    
    # 在 2007 控制器向量中对应的坐标；
    lumi05_mouth_without_R_ctrl_indices: List[int] = field(default_factory=lambda: LUMI05_MOUTH_WITHOUT_R_CTRL_INDICES)
    lumi05_eye_without_R_ctrl_indices: List[int] = field(default_factory=lambda: LUMI05_EYE_WITHOUT_R_CTRL_INDICES)

    lumi05_mouth_L_ctrl_indices: List[int] = field(default_factory=lambda: LUMI05_MOUTH_L_CTRL_INDICES)

    lumi05_eye_L_ctrl_indices: List[int] = field(default_factory=lambda: LUMI05_EYE_L_CTRL_INDICES)

    lumi05_mouth_R_ctrl_indices: List[int] = field(default_factory=lambda: LUMI05_MOUTH_R_CTRL_INDICES)

    lumi05_eye_R_ctrl_indices: List[int] = field(default_factory=lambda: LUMI05_EYE_R_CTRL_INDICES)

    # collate config
    max_feature_size: Optional[int] = None
    pad_feature: bool = True
    random_crop: bool = False

    norm_ori_max: List[float] = field(default_factory=lambda: NORM_ORI_MAX)
    norm_ori_min: List[float] = field(default_factory=lambda: NORM_ORI_MIN)
    norm_tgt_max: float = NORM_TGT_MAX
    norm_tgt_min: float = NORM_TGT_MIN
    norm_map_scale: List[float] = field(default_factory=lambda: NORM_MAP_SCALE)
    norm_remap_scale: List[float] = field(default_factory=lambda: NORM_REMAP_SCALE)
    
    
    
class StaticFeatureDataset(Dataset):
    """
    
    feature_rate: 表明的不是 target 的 feature rate，而是加载数据的原始 feature rate；
     如果 feature rate 和 label rate 不一致，会将其采样到和 label rate 一致； 
     因此如果提前对 feature 做了 resample，那么这里的 feature rate 也应该与 resample 后的 feature 保持一致；
    """

    def __init__(
        self,
        static_audio_feature_manifest_or_list: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        ctrl_manifest_or_list: Union[str, Path, List[Union[str, Path]]] = None,
        feature_rate: Optional[int] = 50,
        label_rate: Optional[int] = 60, 
        max_keep_feature_size: Optional[int] = None,
        min_keep_feature_size: Optional[int] = None, 
        align_length: bool = True
    ):

        self.audio_feature_path_list, self.audio_feature_size_list = load_sequence_data(manifest_or_list=static_audio_feature_manifest_or_list,
                                                                max_keep_sample_size=max_keep_feature_size, min_keep_sample_size=min_keep_feature_size)
        self.ctrl_path_list, self.ctrl_size_list = load_sequence_data(manifest_or_list=ctrl_manifest_or_list)

        self.need_resample = False
        if feature_rate is not None and label_rate is not None:
            # TODO 这里我们先实验简单的做法，例如插值方法，默认要么提前使用 resample_feature_from_50_to_60 插值好，或者 feature rate 只能是 50；
            if feature_rate != label_rate:
                assert feature_rate == 50 and label_rate == 60
                self.need_resample = True

            verify_label_lengths(audio_size_list=self.audio_feature_size_list, audio_path_list=self.audio_feature_path_list, ctrl_size_list=self.ctrl_size_list, 
                                 ctrl_path_list=self.ctrl_path_list, sample_rate=feature_rate, label_rate=label_rate, tol=1)  # TODO 这里有个问题就是部分控制器标签的长度看起来和实际的音频长度不对等；
        
        self.static_audio_feature_manifest_or_list = static_audio_feature_manifest_or_list
        self.ctrl_manifest_or_list = ctrl_manifest_or_list
        self.feature_rate = feature_rate
        self.label_rate = label_rate
        self.max_keep_feature_size = max_keep_feature_size
        self.min_keep_feature_size = min_keep_feature_size
        self.align_length = align_length

    def get_audio_feature(self, index):
        feature = torch.load(self.audio_feature_path_list[index])
        if self.need_resample:
            feature = resample_feature_from_50_to_60(feature)
        return feature

    def get_ctrl_label(self, index):
        return torch.load(self.ctrl_path_list[index])
    
    def __len__(self):
        return len(self.audio_feature_path_list)
    
    def __getitem__(self, index):
        audio_feature = self.get_audio_feature(index)
        ctrl_label = self.get_ctrl_label(index)

        if self.align_length:
            if len(audio_feature) < len(ctrl_label):
                audio_feature = torch.cat([audio_feature, audio_feature[[-1]].repeat(len(ctrl_label)-len(audio_feature), 1)])
            elif len(audio_feature) > len(ctrl_label):
                audio_feature = audio_feature[:len(ctrl_label)]

        return {"idx": index, "audio_feature": audio_feature, "ctrl_label": ctrl_label}

    def num_tokens(self, index) -> int:
        # 用来 order indices；
        return self.audio_feature_size_list[index]

    def size_list(self):
        # 用来 order indices，collate；
        return self.audio_feature_size_list


class PhnDataset(Dataset):
    def __init__(self, phn_manifest_path) -> None:
        self.phn_path_list, _ = load_sequence_data(phn_manifest_path, load_length=False)

    def __len__(self) -> int:
        return len(self.phn_path_list)
    
    def __getitem__(self, index):
        return {
            "phn_dict": torch.load(self.phn_path_list[index])
        }



def static_feature_phn_post_proces_fn(sample):
    audio_feature = sample["audio_feature"]
    ctrl_label = sample["ctrl_label"]
    phn_dict = sample["phn_dict"]

    if "total_length" in phn_dict:
        phn_length = phn_dict["total_length"]
    else:
        phn_length = sum(phn_dict["frame_length_list"])

    audio_feature_length = len(audio_feature)
    ctrl_label_length = len(ctrl_label)
    min_length = min(audio_feature_length, ctrl_label_length, phn_length)
    
    if audio_feature_length > min_length:
        audio_feature = audio_feature[:min_length]
    
    if ctrl_label_length > min_length:
        ctrl_label = ctrl_label[:min_length]

    if phn_length > min_length:
        diff = phn_length - min_length
        phn_list = phn_dict["phn_list"]
        frame_length_list = phn_dict["frame_length_list"]
        for i in range(len(frame_length_list)-1, -1, -1):
            cur_length = frame_length_list[i]
            new_diff = diff - cur_length
            if new_diff >= 0:
                phn_list.pop()
                frame_length_list.pop()
                diff = new_diff
            else:
                frame_length_list[i] -= diff
                break
            
        phn_dict["phn_list"] = phn_list
        phn_dict["frame_length_list"] = frame_length_list
        assert sum(frame_length_list) == min_length
        phn_dict["total_length"] = min_length
    
    sample["audio_feature"] = audio_feature
    sample["ctrl_label"] = ctrl_label
    sample["phn_dict"] = phn_dict
    return sample


class CombinedFeatureDataset(Dataset):
    """
    将多个已经初始化好的 dataset 融合在一起；
    用户需要自己保证相同 index 下在不同数据集中所取到的数据是相互对应的；
    
    """

    def __init__(self, *datasets, post_process_fn: Optional[Callable]=None) -> None:

        logger.info(f"一共横向合并 {len(datasets)} 个数据集.")

        length = len(datasets[0])
        for i in range(1, len(datasets)):
            if len(datasets[i]) != length:
                raise RuntimeError("数据集之间的长度不相同；")

        self.datasets = datasets
        self.post_process_fn = post_process_fn

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, index):
        sample = {}
        for i in range(len(self.datasets)):
            cur_sample = self.datasets[i][index]
            sample.update(cur_sample)
        sample = self.post_process_fn(sample)
        return sample



class AudioOnlyDataset(Dataset):
    def __init__(self, audio_manifest_or_list, sample_rate: int = 16000, normalize: int = False, raw_audio: bool = False) -> None:
        self.audio_path_list, self.audio_size_list = load_sequence_data(manifest_or_list=audio_manifest_or_list)
        
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.raw_audio = raw_audio

    def get_audio(self, index):
        audio_path = self.audio_path_list[index]
        audio, cur_sample_rate = sf.read(audio_path)
        if not self.raw_audio:
            audio = torch.from_numpy(audio).float()
            audio = self.postprocess(audio, cur_sample_rate)
        return audio

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

    def __len__(self):
        return len(self.audio_path_list)
    
    def __getitem__(self, index):
        audio = self.get_audio(index)
        return {"idx": index, "audio": audio}    
    
    def num_tokens(self, index) -> int:
        # 用来 order indices；
        return self.audio_size_list[index]

    def size_list(self):
        # 用来 order indices，collate；
        return self.audio_size_list





class ConstantTokenBatchSampler:
    """
    该类主要负责：
        1. filter by size，如果设置，则对数据集再次进行筛选；
        2. order indices 或者 bucket order indices，对数据按照长度进行排序或者分成几个 ``桶``，每个桶中按照长度进行排序；
        3. constant token batch，一个 batch 中数据长度尽可能一样，并且按照加起来的总长度决定一个 batch 的大小；
            这一步整体的思想就是先按照长度排序，然后分桶；
            如果随机，那么需要 a. 打乱桶之间的顺序；b. 打乱桶内数据的顺序；（如果不随机，那么实际上分桶也没有意义；）
            然后每次 iter 时，将所有桶的数据连在一起，然后按照一个 batch 的 token 数量打包成一个个 batch；

    *** 该 batch sampler 在 multi gpu 中需要进行重写，不过并不复杂，主要的操作在于按照进程数量对数据集进行切片；

    该类用于替换 pytorch dataloader 中原本的 BatchSampler，对于一个 batch sampler 来说，需要考虑以下问题：
        1. 该类是一个可迭代对象，需要实现 __iter__ 方法；
        2. 需要考虑在遍历过程中再次调用 iter 方法是否需要重置该类的内部状态，然后返回一个全新的循环；（通常来讲每次调用 iter 都返回一个全新的循环；）
        3. 需要考虑每次 iter 时是否再对数据的遍历顺序进行随机地打乱；
    """
    def __init__(
        self,
        size_list: List[int],
        one_batch_total_tokens: int,
        filter_size: Optional[int] = None,
        num_buckets: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        dataset_name: str = 'tmp',
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = True,
        **kwargs
    ):
        logger.info(f"Use ``ConstantTokenBatchSampler``, current dataset is {dataset_name}.")

        self.size_list = size_list
        self.one_batch_total_tokens = one_batch_total_tokens
        self.filter_size = filter_size
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        self._seed = seed
        self.dataset_name = dataset_name
        self.epoch = kwargs.get('epoch', 0)

        self.size_list_with_indices = [(idx, w) for idx, w in enumerate(size_list)]
        self.size_list_with_indices = sorted(self.size_list_with_indices, key=lambda x: x[1])

        if filter_size is not None:
            filter_size = min(filter_size, one_batch_total_tokens)
        else:
            filter_size = one_batch_total_tokens

        self.size_list_with_indices, longer_num = self.filter_by_size(self.size_list_with_indices, filter_size, has_sorted=True)
        if longer_num > 0:
            logger.warning(f"Dataset {dataset_name} has {longer_num} samples whose lengths are longer than "
                        f"one_batch_total_tokens: {one_batch_total_tokens} or filter_size: {filter_size}.")

        # 提前分桶；
        if self.shuffle and num_buckets is not None and num_buckets > 1:
            each_bucket_num = len(self.size_list_with_indices) // num_buckets
            indices_buckets = []
            for i in range(num_buckets):
                indices_buckets.append(self.size_list_with_indices[i*each_bucket_num: (i+1)*each_bucket_num])
            indices_buckets[-1].extend(self.size_list_with_indices[(i+1)*each_bucket_num: ])
        else:
            indices_buckets = [self.size_list_with_indices]
        self.indices_buckets = indices_buckets
        self.batches = self.batchify()

        self.during_iter = False
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.set_distributed()


    @staticmethod
    def filter_by_size(size_list_with_indices, filter_size, has_sorted=False):
        tmp_list = []
        longer_num = 0
        if not has_sorted:
            for sample in size_list_with_indices:
                if sample[1] <= filter_size:
                    tmp_list.append(sample)
                else:
                    longer_num += 1
        else:
            for ii in range(len(size_list_with_indices)-1, -1, -1):
                if size_list_with_indices[ii][1] <= filter_size:
                    break
                else:
                    longer_num += 1
            tmp_list = size_list_with_indices[:ii+1]
        return tmp_list, longer_num

    def set_epoch(self, epoch):
        self.epoch = epoch

    @property
    def seed(self) -> int:
        return abs(self._seed + self.epoch)

    def batchify(self):
        if self.shuffle:
            rng = np.random.default_rng(seed=self.seed)
            rng.shuffle(self.indices_buckets)
            for bucket in self.indices_buckets:
                rng.shuffle(bucket)

        batches = []
        cur_batch_max_length = -1
        cur_idx = 0
        cur_batch = []
        indices = list(chain(*self.indices_buckets))

        while cur_idx < len(indices):
            sample_idx, sample_length = indices[cur_idx]
            max_length = max(cur_batch_max_length, sample_length)
            if max_length * (len(cur_batch) + 1) <= self.one_batch_total_tokens:
                cur_batch.append(sample_idx)
            else:
                batches.append(cur_batch)
                cur_batch = [sample_idx]
            cur_batch_max_length = max_length
            cur_idx += 1
        batches.append(cur_batch)

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(batches)

        # 将 token 数量最多的 batch、单个数据长度最长的 batch 以及 batch size 最大的 batch 放在最前面，从而提前发现 OOM；
        max_tokens_batch_idx = np.argmax([sum(self.size_list[w] for w in _batch) for _batch in batches])
        max_length_batch_idx = np.argmax([max(self.size_list[w] for w in _batch) for _batch in batches])
        max_size_batch_idx = np.argmax(map(len, batches))
        for idx in {max_tokens_batch_idx, max_length_batch_idx, max_size_batch_idx}:
            batch = batches.pop(idx)
            batches.insert(0, batch)

        logger.info(f"Dataset {self.dataset_name} uses total {len(indices)} samples and is packed into {len(batches)} batches.")
        return batches

    def __iter__(self):

        self.during_iter = True
        for batch in self.batches[self.rank::self.num_replicas]:
            yield batch
        self.during_iter = False

        if self.shuffle:
            self.batches = self.batchify()
            self.set_distributed()

    def __len__(self):
        return len(self.batches) // self.num_replicas
    

    def set_distributed(self):
        """
        进行分布式的设置，主要作用在于因为这里我们直接重写了 BatchSampler，因此在 DDP 训练时直接一步到位，在 batchsampler 时直接对不同rank的数据进行划分；
        
        按照最简单的设置，只需要指定 world_size 和 当前卡的 rank 即可；
        """
        assert self.during_iter is False, "The batchsampler is now during iteration!"
        if self.drop_last:
            self.batches = self.batches[: len(self.batches) // self.num_replicas * self.num_replicas]



def compute_audio_frame_length_wav2vec2(hin, kernel=(10, 3, 3, 3, 3, 2, 2), stride=(5, 2, 2, 2, 2, 2, 2)):
    for i in range(len(kernel)):
        hin = (hin - kernel[i]) // stride[i] + 1
    return hin


class AudioWav2Vec2Collater:
    def __init__(self, audio_feature_extractor) -> None:
        self.audio_feature_extractor = audio_feature_extractor
    
    def __call__(self, batch):

        sample_list = [s for s in batch if s["audio"] is not None]
        if len(sample_list) == 0:
            return {}

        audio_list = [s["audio"] for s in sample_list]
        audio_input = self.audio_feature_extractor(audio_list, padding=True, return_tensors="pt", sampling_rate=self.audio_feature_extractor.sampling_rate)

        frame_lengths = [compute_audio_frame_length_wav2vec2(len(w)) for w in audio_list]

        idx_list = [s["idx"] for s in sample_list]
        collated_batch = {
            "idx": idx_list,
            "audio_input": audio_input,
            "frame_lengths": frame_lengths
        }

        return collated_batch

 
def pad_audio_fn(
    audio_list,
    max_sample_size: int,
    pad_audio: bool,
    random_crop: bool,
):
    audio_size_list = [len(s) for s in audio_list]

    if pad_audio:
        audio_size = min(max(audio_size_list), max_sample_size)
    else:
        audio_size = min(min(audio_size_list), max_sample_size)

    real_audio_size_list = []
    collated_audios = audio_list[0].new_zeros(len(audio_list), audio_size)
    # wav2vec2 中预训练 base 和 large 都没有使用 padding mask（指加载数据时，但实际上 fairseq 使用了 require len multiple of），而在微调时使用；
    #  fairseq 实现的 hubert 本身则全部默认使用 padding mask；
    # 这里我们按照 transformers 的设定，1 表示需要 attend，0 表示不需要，因此 0 表示该位置是 pad；需要在实现模型的时候考虑到这一点；
    padding_mask = torch.BoolTensor(collated_audios.shape).fill_(True)
    audio_start_list = [0 for _ in range(len(collated_audios))]
    for i, audio in enumerate(audio_list):
        diff = len(audio) - audio_size
        if diff == 0:
            collated_audios[i] = audio
            real_audio_size_list.append(len(audio))
        elif diff < 0:
            assert pad_audio is True
            collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
            padding_mask[i, diff:] = False
            real_audio_size_list.append(len(audio))
        else:
            start, end = 0, audio_size
            if random_crop:
                start = np.random.randint(0, diff + 1)
                end = start + audio_size
            collated_audios[i] = audio[start: end]
            audio_start_list[i] = start
            real_audio_size_list.append(audio_size)

    real_audio_size_list = torch.LongTensor(real_audio_size_list)
    return collated_audios, padding_mask, audio_size, audio_start_list, real_audio_size_list


def pad_discrete_label_fn(
    label_list,
    frame_size,
    frame_start_list,
    pad_token_id: int = 0
):
    label_list = [torch.LongTensor(t[s: s + frame_size]) for t, s in zip(label_list, frame_start_list)]
    lengths = torch.LongTensor([len(t) for t in label_list])
    ntokens = lengths.sum().items()
    collated_labels = label_list[0].new_full((len(label_list), frame_size), pad_token_id)
    for i, label in enumerate(label_list):
        collated_labels[i, :len(label)] = label
    return collated_labels, lengths, ntokens



def random_pad_feature_fn(
    feature_list,
    max_feature_size: int,
    pad_feature: bool,
    random_crop: bool,
):
    feature_size_list = [len(s) for s in feature_list]

    if pad_feature:
        feature_size = min(max(feature_size_list), max_feature_size)
    else:
        feature_size = min(min(feature_size_list), max_feature_size)

    hidden_size = feature_list[0].size(-1)
    collated_features = feature_list[0].new_zeros(len(feature_list), feature_size, hidden_size)
    padding_mask = torch.BoolTensor(size=(len(feature_list), feature_size)).fill_(True)
    feature_start_list = [0 for _ in range(len(collated_features))]
    for i, feature in enumerate(feature_list):
        diff = len(feature) - feature_size
        if diff == 0:
            collated_features[i] = feature
        elif diff < 0:
            assert pad_feature is True
            collated_features[i] = torch.cat([feature, feature.new_full((-diff, hidden_size), 0.0)])
            padding_mask[i, diff:] = False
        else:
            start, end = 0, feature_size
            if random_crop:
                start = np.random.randint(0, diff + 1)
                end = start + feature_size
            collated_features[i] = feature[start: end]
            feature_start_list[i] = start

    return collated_features, padding_mask, feature_size, feature_start_list



def directly_pad_feature_fn(
    feature_list,
    feature_size,
    feature_start_list,
):
    """
    feature_list: [ [N1, H], [N2, H], ... ]
    """
    feature_list = [w[s: s + feature_size] for w, s in zip(feature_list, feature_start_list)]
    lengths = torch.LongTensor([len(t) for t in feature_list])
    ntokens = lengths.sum().item()
    collated_features = feature_list[0].new_zeros((len(feature_list), feature_size, feature_list[0].size(-1)))
    for i, feature in enumerate(feature_list):
        collated_features[i, :len(feature)] = feature
    return collated_features, lengths, ntokens






def pad_phn_fn(
    phn_dict_list,
    feature_size=None,
    directly_pad: bool = True,
    flatten_pad: bool = True,
    phn_padding_idx: int = 0,
    feature_start_list=None,
):
    """
    再使用该函数前已经默认每一个 sample 的 phn 的长度已经提前和 feature 的长度对齐；

    如果需要截断的话，先使用 frame_length_list 扩充 phn_list，然后再重新生成 frame_length_list；

    directly_pad 仅在不需要（可能）截断时有效，即原本 collater 的 max_feature_size 为 None，因此只需要 pad 到当前 batch 的最长 sample 即可；

    not flatten:
        phn_list: [
            [a, b, c, d],
            [a, b, 0, 0],
            [a, 0, 0, 0]
        ],
        frame_length_list: [
            [4, 3, 2, 1],
            [6, 7, 0, 0],
            [3, 0, 0, 0]
        ]
    """

    collated_phns = None
    collated_phn_lists = None
    collated_frame_length_lists = None
    padding_mask = None
    
    if not directly_pad:
        raise NotImplementedError("目前不支持截断的 collate 方式。")
    else:
        # directly_pad 为 True 时，因为保证是没有截断，因此 faeture size 就是当前batch中的 flatten 后最长的 sample 的长度；
        if flatten_pad:
            collated_phns = torch.zeros((len(phn_dict_list), feature_size)).long().fill_(phn_padding_idx)
            for i in range(len(phn_dict_list)):
                phn_dict = phn_dict_list[i]
                phn_list = torch.LongTensor(phn_dict["phn_list"])
                frame_length_list = torch.LongTensor(phn_dict["frame_length_list"])
                total_length = phn_dict["total_length"]
                flatten_phns = torch.repeat_interleave(phn_list, frame_length_list)
                collated_phns[i][:total_length] = flatten_phns
            # 因为这里需要的其他东西在前面已经得到了，例如 padding mask 等，因此这里直接返回 collated_phns；
        else:
            max_phn_list_length = max(len(w["phn_list"]) for w in phn_dict_list)
            collated_phn_lists = torch.zeros((len(phn_dict_list), max_phn_list_length)).long().fill_(phn_padding_idx)
            collated_frame_length_lists = torch.zeros((len(phn_dict_list), max_phn_list_length)).long()
            padding_mask = torch.zeros((len(phn_dict_list), max_phn_list_length)).bool().fill_(False)
            for i in range(len(phn_dict_list)):
                phn_dict = phn_dict_list[i]
                phn_list = torch.LongTensor(phn_dict["phn_list"])
                frame_length_list = torch.LongTensor(phn_dict["frame_length_list"])
                total_length = phn_dict["total_length"]
                collated_phn_lists[i][:len(phn_list)] = phn_list
                collated_frame_length_lists[i][:len(frame_length_list)] = frame_length_list
                padding_mask[i][:len(phn_list)] = True
            
    return {
        "collated_phns": collated_phns,  # flatten
        "collated_phn_lists": collated_phn_lists,
        "collated_frame_length_lists": collated_frame_length_lists,
        "phn_padding_mask": padding_mask
    }




class AudioOnlyCollater:
    def __init__(
        self,
        max_sample_size: int,
        pad_audio: bool,
        random_crop: bool,
    ):
        self.max_sample_size = max_sample_size
        self.pad_audio = pad_audio
        self.random_crop = random_crop

        logger.info(f"AudioOnlyCollater is configured as: \n"
                    f"\tmax_sample_size: {max_sample_size},"
                    f"\tpad_audio: {pad_audio},"
                    f"\trandom_crop: {random_crop}.")

    def __call__(self, batch):
        """
        hubert 的预训练默认是将一个 batch 中的所有音频全部随机裁剪到最短长度；

        :param batch:
        :param max_sample_size: 将全部音频的最大长度限制到这个数；
        :param pad_audio: 是否对一个 batch 中的音频全部 pad 到它们中的最长的长度或者 max_sample_size；如果为 False，那么就将所有音频裁剪
         到这个 batch 中的最短的长度；
        :param random_crop: 如果不 pad 到最长，那么是否在音频中随机裁剪一段，还是说全部默认从 0 开始裁剪到对应长度；
        :param pad_token_id: 用于 pad 标签序列，这里需要和 label_processors 保持一致；
        :return:
        """
        sample_list = [s for s in batch if s["audio"] is not None]
        if len(sample_list) == 0:
            return {}

        audio_list = [s["audio"] for s in sample_list]

        # collate audio;
        collated_audios, padding_mask, audio_size, audio_start_list, real_audio_size_list = pad_audio_fn(
            audio_list=audio_list,
            max_sample_size=self.max_sample_size,
            pad_audio=self.pad_audio,
            random_crop=self.random_crop
        )

        collated_batch = {
            "idx": torch.LongTensor([s['idx'] for s in batch]),
            "net_input": {
                "audios": collated_audios,
                "padding_mask": padding_mask,
            },
        }

        return collated_batch




class RawAudioCollater:

    """
    使用该 collater，默认输入的 feature rate 和 label rate 是相同的，例如提前将 50 Hz 的 hubert 的输出重采样到 60，这一步应该在 dataset 的 __getitem__ 中完成；
    
    """

    def __init__(
        self,
        max_sample_size: int,
        pad_audio: bool,
        random_crop: bool,

        sample_rate: Optional[int] = None,
        label_rate: Optional[int] = None
    ):
        if max_sample_size is None:
            max_sample_size = sys.maxsize

        self.max_sample_size = max_sample_size
        self.pad_audio = pad_audio
        self.random_crop = random_crop

        self.s2f = label_rate / sample_rate

        logger.info(f"StaticFeatureCollater is configured as: \n"
                    f"\tmax_feature_size: {max_sample_size},"
                    f"\tpad_feature: {pad_audio},"
                    f"\trandom_crop: {random_crop},")


    def __call__(self, batch):
        sample_list = [s for s in batch if s["audio"] is not None]
        if len(sample_list) == 0:
            return {}

        audio_list = [s["audio"] for s in sample_list]
        ctrl_label_list = [s['ctrl_label'] for s in sample_list]

        # collate audio;
        collated_audios, padding_mask, audio_size, audio_start_list, real_audio_size_list = pad_audio_fn(
            audio_list=audio_list,
            max_sample_size=self.max_sample_size,
            pad_audio=self.pad_audio,
            random_crop=self.random_crop
        )

        audio_feature_lengths = []
        for _audio_size in real_audio_size_list:
            _feature_size = compute_audio_frame_length_wav2vec2(_audio_size)
            audio_feature_lengths.append(_feature_size)
        audio_feature_lengths = torch.LongTensor(audio_feature_lengths)

        # collate ctrl label;
        if self.pad_audio:
            feature_start_list = [int(round(s * self.s2f)) for s in audio_start_list]
            feature_size = int(round(audio_size * self.s2f))
        else:
            feature_start_list = [int(round(s * self.s2f)) for s in audio_start_list]
            feature_size = int(round(audio_size * self.s2f))
            rem_size_list = [len(t) - s for t, s in zip(ctrl_label_list, feature_start_list)]
            feature_size = min(feature_size, *rem_size_list)

        collated_ctrl_labels, lengths, ntokens = directly_pad_feature_fn(
            feature_list=ctrl_label_list,
            feature_size=feature_size,
            feature_start_list=feature_start_list
        )

        collated_batch = {
            "idx": torch.LongTensor([s['idx'] for s in batch]),
            "net_input": {
                "audios": collated_audios,
                "padding_mask": padding_mask,
                "audio_feature_lengths": audio_feature_lengths
            },
            "label_lengths": lengths,
            "label_ntokens": ntokens,
            "ctrl_labels": collated_ctrl_labels
        }

        return collated_batch





class AudioDiscreteLabelCollater:
    def __init__(
        self,
        max_sample_size: int,
        pad_audio: bool,
        random_crop: bool,
        pad_token_id: int,
        sample_rate: int = 16000,
        label_rate: int = 50
    ):
        self.max_sample_size = max_sample_size
        self.pad_audio = pad_audio
        self.random_crop = random_crop
        self.pad_token_id = pad_token_id
        self.sample_rate = sample_rate
        self.label_rate = sample_rate

        logger.info(f"HubertCollater is configured as: \n"
                    f"\tmax_sample_size: {max_sample_size},"
                    f"\tpad_audio: {pad_audio},"
                    f"\trandom_crop: {random_crop},"
                    f"\tpad_token_id: {pad_token_id},"
                    f"\tsample_rate: {sample_rate},"
                    f"\tlabel_rate: {label_rate}.")

        self.s2f = label_rate / sample_rate

    def __call__(self, batch):
        """
        hubert 的预训练默认是将一个 batch 中的所有音频全部随机裁剪到最短长度；

        :param batch:
        :param max_sample_size: 将全部音频的最大长度限制到这个数；
        :param pad_audio: 是否对一个 batch 中的音频全部 pad 到它们中的最长的长度或者 max_sample_size；如果为 False，那么就将所有音频裁剪
         到这个 batch 中的最短的长度；
        :param random_crop: 如果不 pad 到最长，那么是否在音频中随机裁剪一段，还是说全部默认从 0 开始裁剪到对应长度；
        :param pad_token_id: 用于 pad 标签序列，这里需要和 label_processors 保持一致；
        :return:
        """
        sample_list = [s for s in batch if s["audio"] is not None]
        if len(sample_list) == 0:
            return {}

        audio_list = [s["audio"] for s in sample_list]
        label_list = [s['label'] for s in sample_list]

        # collate audio;
        collated_audios, padding_mask, audio_size, audio_start_list, real_audio_size_list = pad_audio_fn(
            audio_list=audio_list,
            max_sample_size=self.max_sample_size,
            pad_audio=self.pad_audio,
            random_crop=self.random_crop
        )

        # collate label;
        if self.pad_audio:
            frame_start_list = [int(round(s * self.s2f)) for s in audio_start_list]
            frame_size = int(round(audio_size * self.s2f))
        else:
            frame_start_list = [int(round(s * self.s2f)) for s in audio_start_list]
            frame_size = int(round(audio_size * self.s2f))
            rem_size_list = [len(t) - s for t, s in zip(label_list, frame_start_list)]
            frame_size = min(frame_size, *rem_size_list)

        collated_labels, lengths, ntokens = pad_discrete_label_fn(
            label_list=label_list,
            frame_size=frame_size,
            frame_start_list=frame_start_list,
            pad_token_id=self.pad_token_id
        )

        collated_batch = {
            "idx": torch.LongTensor([s['idx'] for s in batch]),
            "net_input": {
                "audios": collated_audios,
                "padding_mask": padding_mask,
            },
            "label_lengths": lengths,
            "label_ntokens": ntokens,
            "labels": collated_labels
        }

        return collated_batch



class StaticFeatureCollater:

    """
    使用该 collater，默认输入的 feature rate 和 label rate 是相同的，例如提前将 50 Hz 的 hubert 的输出重采样到 60，这一步应该在 dataset 的 __getitem__ 中完成；
    
    """

    def __init__(
        self,
        max_feature_size: int,
        pad_feature: bool,
        random_crop: bool,

        phn_directly_pad: bool = True,
        phn_flatten_pad: bool = True,
        phn_padding_idx: int = 0,
    ):
        if max_feature_size is None:
            max_feature_size = sys.maxsize

        self.max_feature_size = max_feature_size
        self.pad_feature = pad_feature
        self.random_crop = random_crop
        self.phn_directly_pad = phn_directly_pad
        self.phn_flatten_pad = phn_flatten_pad
        self.phn_padding_idx = phn_padding_idx

        logger.info(f"StaticFeatureCollater is configured as: \n"
                    f"\tmax_feature_size: {max_feature_size},"
                    f"\tpad_feature: {pad_feature},"
                    f"\trandom_crop: {random_crop},"
                    f"\phn_directly_pad: {phn_directly_pad},"
                    f"\phn_flatten_pad: {phn_flatten_pad},"
                    f"\phn_padding_idx: {phn_padding_idx},")


    def __call__(self, batch):
        sample_list = [s for s in batch if s["audio_feature"] is not None]
        if len(sample_list) == 0:
            return {}

        feature_list = [s["audio_feature"] for s in sample_list]
        ctrl_label_list = [s['ctrl_label'] for s in sample_list]

        for i in range(len(feature_list)):
            assert len(feature_list[i]) == len(ctrl_label_list[i])

        # collate static audio feature;
        collated_features, padding_mask, feature_size, feature_start_list = random_pad_feature_fn(
            feature_list=feature_list,
            max_feature_size=self.max_feature_size,
            pad_feature=self.pad_feature,
            random_crop=self.random_crop
        )

        collated_ctrl_labels, lengths, ntokens = directly_pad_feature_fn(
            feature_list=ctrl_label_list,
            feature_size=feature_size,
            feature_start_list=feature_start_list
        )

        collated_batch = {
            "idx": torch.LongTensor([s['idx'] for s in batch]),
            "net_input": {
                "audio_features": collated_features,
                "padding_mask": padding_mask,
            },
            "label_lengths": lengths,
            "label_ntokens": ntokens,
            "ctrl_labels": collated_ctrl_labels
        }


        # pad phn；
        if "phn_dict" in sample_list[0]:
            phn_dict_list = [s["phn_dict"] for s in sample_list]
            padded_phn_dict = pad_phn_fn(
                phn_dict_list=phn_dict_list,
                feature_size=feature_size,
                directly_pad=self.phn_directly_pad,
                flatten_pad=self.phn_flatten_pad,
                phn_padding_idx=self.phn_padding_idx,
                feature_start_list=feature_start_list
            )
            collated_batch["phn_dict"] = padded_phn_dict

        return collated_batch





