

"""
处理 phoneme 的数据，为e2e生成加入 phoneme embedding 做准备；

"""

import logging
import textgrid
from math import ceil
from pathlib import Path


import torch

from utils import _process_bar, SearchFilePath, align_length_with_directly_insert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)



# 根据 ctrl label 的长度来进行对齐；
def compute_alignment_idx(textgrid_path, ctrl_label_length):
    """
    根据 textgrid 的标注结果来生成对齐信息，返回一串list，其中每个值表示当前分割在整体音频中所占的比例；
    """
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
    except:
        logger.error(f'{textgrid_path} 数据无法读取.')
        return None, None, None
    
    frame_rate = 60
    phonemes = tg[0]
    
    total_dur = phonemes.maxTime - phonemes.minTime

    first_phn = phonemes[0]
    last_phn = first_phn.mark
    phn_begin = first_phn.minTime
    phn_end = first_phn.maxTime
    phoneme_list = []
    frame_length_list = []

    for i in range(1, len(phonemes)):
        phoneme_interval = phonemes[i]
        cur_phn = phoneme_interval.mark
        if cur_phn != last_phn:
            phoneme_list.append(last_phn)
            frame_length = int((phn_end - phn_begin) / total_dur * ctrl_label_length + 0.00001)
            frame_length_list.append(frame_length)
            phn_begin = phoneme_interval.minTime
            last_phn = cur_phn
        phn_end = phoneme_interval.maxTime

    phoneme_interval = phonemes[-1]
    cur_phn = phoneme_interval.mark
    phoneme_list.append(cur_phn)
    frame_length = ceil((phn_end - phn_begin) / total_dur * ctrl_label_length)
    frame_length_list.append(frame_length)
    
    phn_length = sum(frame_length_list)
    if phn_length > ctrl_label_length:
        diff = phn_length - ctrl_label_length
        for i in range(len(frame_length_list)-1, -1, -1):
            cur_length = frame_length_list[i]
            new_diff = diff - cur_length
            if new_diff >= 0:
                phoneme_list.pop()
                frame_length_list.pop()
                diff = new_diff
            else:
                frame_length_list[i] -= diff
                break
    elif phn_length < ctrl_label_length:
        flatten_phns = []
        for phn, length in zip(phoneme_list, frame_length_list):
            flatten_phns.extend([phn] * length)
        
        aligned_flatten_phns = align_length_with_directly_insert(flatten_phns, ctrl_label_length)
        aligned_phn_list = []
        aligned_frame_length_list = []
        _phn = aligned_flatten_phns[0]
        _length = 1
        for i in range(1, len(aligned_flatten_phns)):
            _cur_phn = aligned_flatten_phns[i]
            if _cur_phn != _phn:
                aligned_phn_list.append(_phn)
                aligned_frame_length_list.append(_length)
                _phn = _cur_phn
                _length = 1
            else:
                _length += 1
        aligned_phn_list.append(_phn)
        aligned_frame_length_list.append(_length)
        return aligned_phn_list, aligned_frame_length_list

    return phoneme_list, frame_length_list



def read_vocab(vocab_path):
    vocab_dict = {}
    re_vocab_dict = {}
    with open(vocab_path, "r") as f:
        vocab_idx = 0
        for line in f:
            _label = line.rstrip()
            vocab_dict[_label] = vocab_idx
            re_vocab_dict[vocab_idx] = _label
            vocab_idx += 1
    
    return vocab_dict, re_vocab_dict
        


# 这里不生成 manifest，因为有些音频或者 ctrl label 没有；
def prepare_phoneme_indices(phn_dirs, vocab_path, ctrl_label_length_dict, save_dir):
    if not isinstance(phn_dirs, (str, Path)):
        phn_dirs = [phn_dirs]

    vocab_dict, _ = read_vocab(vocab_path)

    search_model = SearchFilePath()
    all_path_list = search_model.find_all_file_paths(
        data_dirs=phn_dirs,
        pattern="alignment.textgrid",
        n_proc=64,
        depth=1
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # root_path = Path("/data/lipsync/xgyang/e2e_data/static_feature/hubert_60")

    with _process_bar("prepare phoneme indices", total=len(all_path_list)) as update:
        for _path in all_path_list:
            _path = Path(_path)
            name = _path.parent.stem
            if name in ctrl_label_length_dict:
                ctrl_label_length = ctrl_label_length_dict[name]
                _phoneme_list, _frame_length_list = compute_alignment_idx(_path, ctrl_label_length)
                
                _phoneme_list = [vocab_dict[w] for w in _phoneme_list]
                _name = _path.parent.name + ".pt"

                # try:
                #     hubert_data = len(torch.load(root_path.joinpath(_name)))
                #     print(f"name, length, hubert length: {_name}, {sum(_frame_length_list)}/{hubert_data}")
                # except:
                #     pass
                total_length = sum(_frame_length_list)
                assert total_length == ctrl_label_length
                torch.save({"phn_list": _phoneme_list, "frame_length_list": _frame_length_list, "total_length": total_length}, save_dir.joinpath(_name))
                update(1)



if __name__ == "__main__":

    # tg_path = "/data/lipsync/xgyang/e2e_data/yingrao/dataproc/phonemes/afraid_000000-afraid-great/alignment.textgrid"
    # compute_alignment_idx(tg_path)

    from align_feature_length import read_ctrl_label_length_dict

    length_dict = read_ctrl_label_length_dict("/data/lipsync/xgyang/e2e_data/ctrl_label_length_dict.txt")

    prepare_phoneme_indices(
        phn_dirs="/data/lipsync/xgyang/e2e_data/yingrao/dataproc/phonemes",
        vocab_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/static_file/vocab.txt",
        ctrl_label_length_dict=length_dict,
        save_dir="/data/lipsync/xgyang/e2e_data/aligned_phonemes"
    )



























