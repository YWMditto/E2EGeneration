

import logging
from typing import Optional, List, Union
from pathlib import Path
import soundfile as sf
from itertools import chain
import numpy as np
from math import ceil

import torch
import torch.nn as nn
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
            size = int(size)
            if max_keep_sample_size is not None and size > max_keep_sample_size:
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
    if not hasattr(resample_feature_from_50_to_60, "InterpolateIndicesMap"):
        resample_feature_from_50_to_60.InterpolateIndicesMap = {}
    InterpolateIndicesMap = resample_feature_from_50_to_60.InterpolateIndicesMap

    computed_indices = []
    for int_idx in interpolate_indices:
        try:
            computed_indices.append(InterpolateIndicesMap[int_idx])
        except KeyError:
            InterpolateIndicesMap[int_idx] = [max(int_idx -2, 0), max(int_idx -1, 0), min(int_idx +1, mapped_length), min(int_idx +2, mapped_length)]
            computed_indices.append(InterpolateIndicesMap[int_idx])
        except Exception as e:
            raise e
    computed_indices = torch.LongTensor(computed_indices)

    interpolated_feature = feature.new_zeros((mapped_length +1, feature.size(1)))
    interpolated_feature[directly_map_indices] = feature

    weights = torch.FloatTensor([1 /6, 1/ 3, 1 / 3, 1 / 6]).unsqueeze(0).unsqueeze(2)
    computed_feature_value = (interpolated_feature[computed_indices] * weights).sum(dim=1)
    interpolated_feature[interpolate_indices] = computed_feature_value

    return interpolated_feature


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
                                 ctrl_path_list=self.ctrl_path_list, sample_rate=feature_rate, label_rate=label_rate, tol=0.1)
        
        self.static_audio_feature_manifest_or_list = static_audio_feature_manifest_or_list
        self.ctrl_manifest_or_list = ctrl_manifest_or_list
        self.feature_rate = feature_rate
        self.label_rate = label_rate
        self.max_keep_feature_size = max_keep_feature_size
        self.min_keep_feature_size = min_keep_feature_size

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
        rank: int = 1,
        drop_last: bool = True,
        **kwargs
    ):
        logger.info(f"Use ``HubertBatchSampler``, current dataset is {dataset_name}.")

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
        elif diff < 0:
            assert pad_audio is True
            collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
            padding_mask[i, diff:] = False
        else:
            start, end = 0, audio_size
            if random_crop:
                start = np.random.randint(0, diff + 1)
                end = start + audio_size
            collated_audios[i] = audio[start: end]
            audio_start_list[i] = start

    return collated_audios, padding_mask, audio_size, audio_start_list


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
        feature_size = max(min(feature_size_list), max_feature_size)
    else:
        feature_size = min(min(feature_size_list), max_feature_size)

    hidden_size = feature_list[0].size(-1)
    collated_features = feature_list[0].new_zeros(len(feature_list), feature_size, hidden_size)
    padding_mask = torch.BoolTensor((len(feature_list), feature_size)).fill_(True)
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
    ntokens = lengths.sum().items()
    collated_features = feature_list[0].new_zeros((len(feature_list), feature_size, feature_list[0].size(-1)))
    for i, feature in enumerate(feature_list):
        collated_features[i, :len(feature)] = feature
    return collated_features, lengths, ntokens




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
        collated_audios, padding_mask, audio_size, audio_start_list = pad_audio_fn(
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
        collated_audios, padding_mask, audio_size, audio_start_list = pad_audio_fn(
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
    ):
        self.max_feature_size = max_feature_size
        self.pad_feature = pad_feature
        self.random_crop = random_crop

        logger.info(f"StaticFeatureCollater is configured as: \n"
                    f"\max_feature_size: {max_feature_size},"
                    f"\pad_feature: {pad_feature},"
                    f"\trandom_crop: {random_crop},")


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

        return collated_batch





