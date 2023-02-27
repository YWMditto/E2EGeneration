
import logging
from pathlib import Path
import re
from copy import deepcopy
from multiprocessing import Pool, RLock
from contextlib import contextmanager
from copy import deepcopy
import time
from tqdm import tqdm
from typing import Union, List, Sequence, Tuple, Dict
import py3nvml.nvidia_smi as nvidia_smi
import torch

logger = logging.getLogger(__file__)



def multi_process_fn(process_fn, data_list, n_proc=2, reduce_fn=None):
    
    logger.info(f"一共处理 {len(data_list)} 个数据.")
    each_process_num = (len(data_list) + n_proc - 1) // n_proc

    all_data_list = []

    for i in range(n_proc):
        all_data_list.append(data_list[i*each_process_num: (i+1)*each_process_num])

    all_data_list[-1].extend(data_list[(i+1)*each_process_num: ])

    # TODO 这里 tqdm 或者说多进程的进度条的显示还是有很多问题；
    with Pool(processes=n_proc, initargs=(RLock(), ), initializer=tqdm.set_lock) as pool:
        jobs = [pool.apply_async(process_fn, args=(all_data_list[w], w)) for w in range(n_proc)]
        res_list = [_job.get() for _job in jobs]
    
    if reduce_fn is not None:
        return reduce_fn(res_list)
    else:
        return res_list


@contextmanager
def _process_bar(task_name, total, pid=0):
    tqdm_text = task_name + "{}".format(pid).zfill(3)
    with tqdm(total=total, desc=tqdm_text, position=pid+1, leave=False) as pbar:
        yield pbar.update
        pbar.clear()


class SearchFilePath:

    def __init__(self) -> None:
        
        self.pattern = None

    def _process(self, _data_dirs, pid=0):
        _all_dirs = deepcopy(_data_dirs)[::-1]
        _all_wav_paths = set()

        while len(_all_dirs):
            cur_dir = _all_dirs.pop()
            cur_dir = Path(cur_dir)
            for _path in cur_dir.iterdir():
                if _path.is_dir():
                    _all_dirs.append(_path)
                elif re.match(self.pattern, _path.name):
                    _all_wav_paths.add(str(_path))
        return _all_wav_paths

    def find_all_file_paths(self, data_dirs, pattern, n_proc=1, depth=1):
        if len(data_dirs) == 0:
            return []

        assert depth > 0

        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]

        self.pattern = pattern

        cur_dirs = deepcopy(data_dirs)
        all_wav_paths = set()
        for _ in range(depth):
            new_cur_dirs = []
            for _data_dir in cur_dirs:
                _data_dir = Path(_data_dir)
                for _data_path in _data_dir.iterdir():
                    _data_path = Path(_data_path)
                    if _data_path.is_dir():
                        new_cur_dirs.append(_data_path)
                    elif re.match(pattern, _data_path.name):
                        all_wav_paths.add(str(_data_path))

            cur_dirs = new_cur_dirs
                
        if len(cur_dirs) > 0:
            rest_wav_paths_list = multi_process_fn(self._process, cur_dirs, n_proc=n_proc, reduce_fn=None)

            for _sub_set in rest_wav_paths_list:
                all_wav_paths.update(_sub_set)

        return list(all_wav_paths)



def check_gpu_available(device: Union[int, List[int]], limit=18 * 1024**3):

    nvidia_smi.nvmlInit()

    if not isinstance(device, Sequence):
        device = [device]

    could_use = True
    for each_device in device:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(each_device)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        if info.free < limit:
            could_use = False
            break

    return could_use



def move_data_to_device(batch, device, dtype=torch.Tensor):
    if isinstance(batch, dtype):
        batch = batch.to(device)
        return batch
    elif isinstance(batch, (List, Tuple)):
        tmp_list = []
        for sub_batch in batch:
            _data = move_data_to_device(sub_batch, device, dtype)
            tmp_list.append(_data)
        batch = tmp_list
    else:
        try:
            for key, value in batch.items():
                batch[key] = move_data_to_device(value, device, dtype)
        except:
            return batch
    return batch





