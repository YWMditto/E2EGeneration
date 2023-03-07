

"""
处理 phoneme 的数据，为e2e生成加入 phoneme embedding 做准备；

"""

import logging
import textgrid
from math import ceil
from pathlib import Path


import torch


from utils import _process_bar, SearchFilePath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)




def compute_alignment_idx(textgrid_path):
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
            frame_length = round((phn_end - phn_begin) * frame_rate + 0.00001)
            frame_length_list.append(frame_length)
            phn_begin = phoneme_interval.minTime
            last_phn = cur_phn
        phn_end = phoneme_interval.maxTime

    phoneme_interval = phonemes[-1]
    cur_phn = phoneme_interval.mark
    phoneme_list.append(cur_phn)
    frame_length = ceil((phn_end - phn_begin) * frame_rate)
    frame_length_list.append(frame_length)
    
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
    
    return vocab_dict, re_vocab_dict
        


# 这里不生成 manifest，因为有些音频或者 ctrl label 没有；
def prepare_phoneme_indices(phn_dirs, vocab_path, save_dir):
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
            _phoneme_list, _frame_length_list = compute_alignment_idx(_path)
            

            _phoneme_list = [vocab_dict[w] for w in _phoneme_list]
            _name = _path.parent.name + ".pt"

            # try:
            #     hubert_data = len(torch.load(root_path.joinpath(_name)))
            #     print(f"name, length, hubert length: {_name}, {sum(_frame_length_list)}/{hubert_data}")
            # except:
            #     pass

            torch.save({"phn_list": _phoneme_list, "frame_length_list": _frame_length_list, "total_length": sum(_frame_length_list)}, save_dir.joinpath(_name))

            update(1)



if __name__ == "__main__":

    # tg_path = "/data/lipsync/xgyang/e2e_data/yingrao/dataproc/phonemes/afraid_000000-afraid-great/alignment.textgrid"
    # compute_alignment_idx(tg_path)

    prepare_phoneme_indices(
        phn_dirs="/data/lipsync/xgyang/e2e_data/yingrao/dataproc/phonemes",
        vocab_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/vocab.txt",
        save_dir="/data/lipsync/xgyang/e2e_data/phoneme"
    )



























