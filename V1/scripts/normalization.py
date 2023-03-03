

"""

对标签进行归一化；



"""

import sys
sys.path.extend([".", ".."])

import logging
from pathlib import Path


import torch

from data_prepare import StaticFeatureDatasetConfig, norm_encode
from utils import _process_bar, SearchFilePath


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)




def normalize_ctrl_label(ctrl_label_dirs, save_dir, tgt_max, tgt_min):
    """
    这个函数做了两件事情：
    1. 根据实际的 mouth 和 eye 的 indices，直接抽取出实际的 66 个控制器标签存储；
    2. 对抽取出来的特征的每一维进行归一化，归一到
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    if isinstance(ctrl_label_dirs, (str, Path)):
        ctrl_label_dirs = [ctrl_label_dirs]

    search_model = SearchFilePath()
    all_path_list = search_model.find_all_file_paths(
        data_dirs=ctrl_label_dirs,
        pattern=".+\.pt",
        n_proc=64,
        depth=1
    )

    statuc_feature_dataset_config = StaticFeatureDatasetConfig()
    mouth_indices = statuc_feature_dataset_config.lumi05_mouth_without_R_ctrl_indices
    eye_indices = statuc_feature_dataset_config.lumi05_eye_without_R_ctrl_indices
    total_indices = mouth_indices + eye_indices
    total_indices_num = len(total_indices)

    logger.info(f"total indices num: {total_indices_num}, mouth / eye indices num: {len(mouth_indices)} / {len(eye_indices)}.")

    # 统计统计量，ori_max, ori_min, tgt_max, tgt_min；
    ori_max = torch.FloatTensor([-100 for _ in range(total_indices_num)])
    ori_min = torch.FloatTensor([100 for _ in range(total_indices_num)])

    with _process_bar("normalize ctrl label", total=len(all_path_list)) as update:
        for _path in all_path_list:
            _data = torch.load(_path)
            _used_data = _data[:, total_indices]
            ori_max = torch.max(ori_max, _used_data.max(dim=0)[0])
            ori_min = torch.min(ori_min, _used_data.min(dim=0)[0])

            new_path = save_dir.joinpath(Path(_path).name)
            torch.save(_used_data, new_path)

            update(1)
    
    logger.info(f"ori max: {ori_max}.")
    logger.info(f"ori min: {ori_min}.")

    map_scale = (tgt_max - tgt_min) / (ori_max - ori_min)
    logger.info(f"map scale: {map_scale}.")

    # normalize;

    with _process_bar("normalize ctrl label", total=len(all_path_list)) as update:
        for _path in all_path_list:
            _data = torch.load(_path)
            _used_data = _data[:, total_indices]

            _used_data = norm_encode(_used_data, ori_min=ori_min, tgt_min=tgt_min, map_scale=map_scale)

            new_path = save_dir.joinpath(Path(_path).name)
            torch.save(_used_data, new_path)

            update(1)



if __name__ == "__main__":

    normalize_ctrl_label(
        ctrl_label_dirs="/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05",
        save_dir="/data/lipsync/xgyang/e2e_data/normalized_extracted_ctrl_labels",
        tgt_max=1.,
        tgt_min=-1.
    )












