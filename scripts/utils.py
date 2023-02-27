
from pathlib import Path

import torch
import numpy as np




def transform_pt2txt(path_or_list, save_path):

    if isinstance(path_or_list, (str, Path)):
        path_or_list = [path_or_list]
    
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    for _path in path_or_list:
        _path = Path(_path)
        _data = torch.load(_path)
        _data = _data.numpy()
        np.save(save_path.joinpath(_path.stem), _data)
    


if __name__ == "__main__":
    transform_pt2txt(
        [
            "/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05/daodao_000056-neutral.pt",
            "/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05/sad_000032-sad-little.pt",
            "/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05/surprise_000199-surprise-little.pt",
            "/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05/biaobei_001663-neutral.pt",
        ],
        "/data/lipsync/xgyang/E2EGeneration/tmp_dir/maya_generation"
    )




