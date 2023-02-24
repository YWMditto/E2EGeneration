

import logging
from typing import Optional, List, Union
from pathlib import Path
import soundfile as sf

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


logger = logging.getLogger(__file__)



def load_sequence_data(manifest_or_list, max_keep_sample_size=None, min_keep_sample_size=None):
    if max_keep_sample_size is not None and min_keep_sample_size is not None:
        assert 0 <= min_keep_sample_size < max_keep_sample_size

    path_list = []
    size_list = []

    longer_num = 0
    shorter_num = 0
    with open(manifest_or_list, 'r') as f:
        for line in f:
            _path, size = line.rstrip().split('\t')
            if max_keep_sample_size is not None and (size := int(size)) > max_keep_sample_size:
                longer_num += 1
            elif min_keep_sample_size is not None and size < min_keep_sample_size:
                shorter_num += 1
            else:
                path_list.append(str(_path))
                size_list.append(size)
    
    logger.info(
        (
            f"max_keep={max_keep_sample_size}, min_keep={min_keep_sample_size}, "
            f"loaded {len(path_list)}, skipped {shorter_num} short and {longer_num} long, "
            f"longest-loaded={max(size_list)}, shortest-loaded={min(size_list)}"
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



class RawAudioDataset(Dataset):

    """
    加载训练数据，音频和实际的控制器数据都是在实际使用时才会读取；
    
    可以指定是加载原始音频还是加载从预训练模型中抽取出来的音频特征；
        如果是加载固定的音频特征，那么注意此时的 sample_rate 表示的实际上是 label_rate 的意思，即表示 1 s内多少音频特征；


    Note: 控制器总体的数量很多（2007），但是在实际训练的时候我们只需要其中的一部分，例如 eye 和 mouth，这里数据集不需要管这些操作，如果需要在每次拿到数据的时候再抽取，
     那么需要在实际训练的时候进行额外的处理；
            
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


class StaticFeatureDataset(Dataset):
    """
    
    feature_rate: 表明的不是 target 的 feature rate，而是加载数据的原始 feature rate；
     如果 feature rate 和 label rate 不一致，会将其采样到和 label rate 一致； 
    
    """


    def __init__(
        self,
        static_audio_feature_manifest_or_list: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        ctrl_manifest_or_list: Union[str, Path, List[Union[str, Path]]] = None,
        feature_rate: Optional[int] = 50,
        label_rate: Optional[int] = 60, 
        max_keep_feature_size: Optional[int] = None,
        min_keep_feature_size: Optional[int] = None, 
    ):

        self.audio_feature_path_list, self.audio_feature_size_list = load_sequence_data(manifest_or_list=static_audio_feature_manifest_or_list, 
                                                                max_keep_sample_size=max_keep_feature_size, min_keep_sample_size=min_keep_feature_size)
        self.ctrl_path_list, self.ctrl_size_list = load_sequence_data(manifest_or_list=ctrl_manifest_or_list)

        if feature_rate is not None and label_rate is not None:
            verify_label_lengths(audio_size_list=self.audio_feature_size_list, audio_path_list=self.audio_feature_path_list, ctrl_size_list=self.ctrl_size_list, 
                                ctrl_path_list=self.ctrl_path_list, sample_rate=feature_rate, label_rate=label_rate, tol=0.1)
        
        self.static_audio_feature_manifest_or_list = static_audio_feature_manifest_or_list
        self.ctrl_manifest_or_list = ctrl_manifest_or_list
        self.feature_rate = feature_rate
        self.label_rate = label_rate
        self.max_keep_feature_size = max_keep_feature_size
        self.min_keep_feature_size = min_keep_feature_size

    def get_audio_feature(self, index):
        return torch.load(self.audio_feature_path_list[index])

    def get_ctrl_label(self, index):
        return torch.load(self.ctrl_path_list[index])
    
    def __len__(self):
        return len(self.audio_feature_path_list)
    
    def __getitem__(self, index):
        audio_feature = self.get_audio_feature(index)
        ctrl_label = self.get_ctrl_label(index)
        return {"idx": index, "audio_feature": audio_feature, "ctrl_label": ctrl_label}

    def num_tokens(self, index) -> int:
        # 用来 order indices；
        return self.audio_feature_size_list[index]

    def size_list(self):
        # 用来 order indices，collate；
        return self.audio_feature_size_list









