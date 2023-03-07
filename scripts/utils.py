import sys
sys.path.append(".")
sys.path.append("..")

from pathlib import Path

import torch
import numpy as np

from V1.scripts.utils import SearchFilePath, _process_bar



def transform_pt2npy(path_or_list, save_path):

    if isinstance(path_or_list, (str, Path)):
        path_or_list = [path_or_list]
    
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    for _path in path_or_list:
        _path = Path(_path)
        _data = torch.load(_path)
        _data = _data.numpy()
        np.save(save_path.joinpath(_path.stem), _data)
    


def transform_pt2npy_dir(path_dir_or_list, save_path):

    if isinstance(path_dir_or_list, (str, Path)):
        path_dir_or_list = [path_dir_or_list]

    search_model = SearchFilePath()
    all_path_list = search_model.find_all_file_paths(
        data_dirs=path_dir_or_list,
        pattern=".+\.pt",
        n_proc=64,
        depth=1
    )

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    with _process_bar("transform pt2npy", total=len(all_path_list)) as update:
        for _path in all_path_list:
            _path = Path(_path)
            _data = torch.load(_path)
            _data = _data.numpy()
            np.save(save_path.joinpath(_path.stem), _data)

            update(1)




def pull_music_to_local(ctrl_pt_path, music_dirs, save_dir):
    ...




if __name__ == "__main__":
    # transform_pt2npy(
    #     [
    #         "/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05/afraid_000005-afraid-great.pt",
    #         "/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05/biaobei_003066-neutral.pt",
    #         "/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05/cute_000001-cute.pt",
    #         "/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05/daodao_000014-neutral.pt",
    #         "/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05/happy_000022-happy-great.pt",
    #         "/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05/sad_000072-sad-little.pt",
    #         "/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05/surprise_000128-surprise-great.pt",
    #     ],
    #     "/data/lipsync/xgyang/E2EGeneration/tmp_dir/maya_generation"
    # )

    # transform_pt2npy_dir(
    #     path_dir_or_list="/data/lipsync/xgyang/E2EGeneration/V1/tmp_dir/evaluate_generation/model_1/v1/only_left/epoch=21-step=17556-validate_loss=0.60",
    #     save_path="/data/lipsync/xgyang/E2EGeneration/V1/tmp_dir/evaluate_generation/model_1/v1/only_left/epoch=21-step=17556-validate_loss=0.60_npy"
    # )

    # transform_pt2npy_dir(
    #     path_dir_or_list="/data/lipsync/xgyang/E2EGeneration/cur_eege/tmp_dir/output/checkpoint_e0020_val1.1691/inference",
    #     save_path="/data/lipsync/xgyang/E2EGeneration/tmp_dir/cur_eege"
    # )

    transform_pt2npy_dir(
        path_dir_or_list="/data/lipsync/xgyang/E2EGeneration/V1/tmp_dir/evaluate_generation/model_1/v1/epoch=99-step=79800-validate_loss=0.11",
        save_path="/data/lipsync/xgyang/E2EGeneration/V1/tmp_dir/evaluate_generation/model_1/v1/epoch=99-step=79800-validate_loss=0.11_npy"
    )



