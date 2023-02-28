

import logging
from pathlib import Path
import soundfile as sf
import random

import torch

from utils import multi_process_fn, _process_bar, SearchFilePath

logger = logging.getLogger(__file__)




def resample_audios(wav_manifest_path, manifest_save_path, audio_save_dir, tgt_sample_rate=16000):

    import librosa

    manifest_save_path = Path(manifest_save_path)
    manifest_save_path.parent.mkdir(exist_ok=True, parents=True)

    audio_save_dir = Path(audio_save_dir)
    audio_save_dir.mkdir(exist_ok=True, parents=True)

    save_f = open(manifest_save_path, "w")

    with open(wav_manifest_path, "r") as f:
        for line in f:
            _path, _frame = line.rstrip().split("\t")
            _audio, _sample_rate = sf.read(_path)
            if _sample_rate != tgt_sample_rate:
                _audio = librosa.resample(_audio, orig_sr=_sample_rate, target_sr=tgt_sample_rate)
                new_path = audio_save_dir.joinpath(Path(_path).name)
                sf.write(new_path, _audio, samplerate=tgt_sample_rate)
                write_line = str(new_path) + "\t" + str(len(_audio))
                print(write_line, file=save_f)
            
    

def prepare_wav_manifest(path_dirs, save_path):
    if isinstance(path_dirs, (str, Path)):
        path_dirs = [path_dirs]

    search_model = SearchFilePath()
    all_path_list = search_model.find_all_file_paths(
        data_dirs=path_dirs,
        pattern=".+\.wav",
        n_proc=64,
        depth=1
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    save_f = open(save_path, 'w') 

    for _path in all_path_list:
        frames = sf.info(_path).frames
        line = str(_path) + "\t" + str(frames)
        print(line, file=save_f)

    save_f.close()


def prepare_static_feature_manifest(path_dirs, save_path):
    if isinstance(path_dirs, (str, Path)):
        path_dirs = [path_dirs]

    search_model = SearchFilePath()
    all_path_list = search_model.find_all_file_paths(
        data_dirs=path_dirs,
        pattern=".+\.pt",
        n_proc=64,
        depth=1
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    save_f = open(save_path, 'w') 

    with _process_bar("prepare static feature manifest", total=len(all_path_list), pid=0) as update:
        for _path in all_path_list:
            feature = torch.load(_path)
            line = str(_path) + "\t" + str(len(feature))
            print(line, file=save_f)

            update(1)

    save_f.close()


def prepare_both_manifest(audio_path_dirs, ctrl_path_dirs, audio_save_path, ctrl_save_path):
    import torch

    def _prepare(path_dirs, save_path, pattern, depth):
        if isinstance(path_dirs, (str, Path)):
            path_dirs = [path_dirs]

        search_model = SearchFilePath()
        all_path_list = search_model.find_all_file_paths(
            data_dirs=path_dirs,
            pattern=pattern,
            n_proc=64,
            depth=depth
        )

        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        save_f = open(save_path, 'w') 
        return all_path_list, save_f

    audio_path_list, audio_f = _prepare(audio_path_dirs, audio_save_path, ".+\.wav", 1)
    ctrl_path_list, ctrl_f = _prepare(ctrl_path_dirs, ctrl_save_path, ".+\.pt", 1)

    def _conv_name_dict(path_list) -> dict:
        tmp_dict = {}
        for _path in path_list:
            _path = Path(_path)
            tmp_dict[_path.stem] = str(_path)
        return tmp_dict
    
    audio_dict = _conv_name_dict(audio_path_list)
    ctrl_dict = _conv_name_dict(ctrl_path_list)

    audio_set = set(audio_dict)
    ctrl_set = set(ctrl_dict)
    audio_set &= ctrl_set

    with _process_bar("write manifest", total=len(audio_set)) as update:
        for _name in audio_set:
            _audio_path = audio_dict[_name]
            _ctrl_path = ctrl_dict[_name]
            frames = sf.info(_audio_path).frames
            audio_line = str(_audio_path) + "\t" + str(frames)
            print(audio_line, file=audio_f)

            _data = torch.load(_ctrl_path)
            length = _data.size(0)
            ctrl_line = str(_ctrl_path) + "\t" + str(length)
            print(ctrl_line, file=ctrl_f)

            update(1)

    audio_f.close()
    ctrl_f.close()




def prepare_static_feature_both_manifest(feature_path_dirs, ctrl_path_dirs, feature_save_path, ctrl_save_path):
    import torch

    def _prepare(path_dirs, save_path, pattern, depth):
        if isinstance(path_dirs, (str, Path)):
            path_dirs = [path_dirs]

        search_model = SearchFilePath()
        all_path_list = search_model.find_all_file_paths(
            data_dirs=path_dirs,
            pattern=pattern,
            n_proc=64,
            depth=depth
        )

        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        save_f = open(save_path, 'w') 
        return all_path_list, save_f

    feature_path_list, feature_f = _prepare(feature_path_dirs, feature_save_path, ".+\.pt", 1)
    ctrl_path_list, ctrl_f = _prepare(ctrl_path_dirs, ctrl_save_path, ".+\.pt", 1)

    def _conv_name_dict(path_list) -> dict:
        tmp_dict = {}
        for _path in path_list:
            _path = Path(_path)
            tmp_dict[_path.stem] = str(_path)
        return tmp_dict
    
    feature_dict = _conv_name_dict(feature_path_list)
    ctrl_dict = _conv_name_dict(ctrl_path_list)

    feature_set = set(feature_dict)
    ctrl_set = set(ctrl_dict)
    feature_set &= ctrl_set

    with _process_bar("write manifest", total=len(feature_set)) as update:
        for _name in feature_set:
            _feature_path = feature_dict[_name]
            _ctrl_path = ctrl_dict[_name]
            _feature = torch.load(_feature_path)
            feature_length = _feature.size(0)
            feature_line = str(_feature_path) + "\t" + str(feature_length)
            print(feature_line, file=feature_f)

            _data = torch.load(_ctrl_path)
            length = _data.size(0)
            ctrl_line = str(_ctrl_path) + "\t" + str(length)
            print(ctrl_line, file=ctrl_f)

            update(1)

    feature_f.close()
    ctrl_f.close()


def split_validate_static_feature(feature_manifest_path, ctrl_manifest_path, save_dir, validate_p=30):

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    def read_txt(file_path):
        with open(file_path, 'r') as f:
            data_list = [line.rstrip() for line in f]
        return data_list

    feature_list = read_txt(feature_manifest_path)
    ctrl_list = read_txt(ctrl_manifest_path)

    assert len(feature_list) == len(ctrl_list)

    if isinstance(validate_p, float):
        assert 0 < validate_p < 1
        validate_p = int(len(feature_list) * validate_p)
    
    validate_indices = random.sample(range(len(feature_list)), validate_p)

    train_feature_mainfest_path = Path(feature_manifest_path)
    train_feature_mainfest_path = train_feature_mainfest_path.parent.joinpath(train_feature_mainfest_path.stem + "_train" + train_feature_mainfest_path.suffix)
    validate_feature_mainfest_path = Path(feature_manifest_path)
    validate_feature_mainfest_path = validate_feature_mainfest_path.parent.joinpath(validate_feature_mainfest_path.stem + "_validate" + validate_feature_mainfest_path.suffix)

    train_ctrl_mainfest_path = Path(ctrl_manifest_path)
    train_ctrl_mainfest_path = train_ctrl_mainfest_path.parent.joinpath(train_ctrl_mainfest_path.stem + "_train" + train_ctrl_mainfest_path.suffix)
    validate_ctrl_mainfest_path = Path(ctrl_manifest_path)
    validate_ctrl_mainfest_path = validate_ctrl_mainfest_path.parent.joinpath(validate_ctrl_mainfest_path.stem + "_validate" + validate_ctrl_mainfest_path.suffix)

    train_feature_f = open(train_feature_mainfest_path, "w")
    validate_feature_f = open(validate_feature_mainfest_path, "w")
    train_ctrl_f = open(train_ctrl_mainfest_path, "w")
    validate_ctrl_f = open(validate_ctrl_mainfest_path, "w")
    
    validate_indices = set(validate_indices)
    for idx, line in enumerate(zip(feature_list, ctrl_list)):
        feature_line, ctrl_line = line
        if idx in validate_indices:
            print(feature_line, file=validate_feature_f)
            print(ctrl_line, file=validate_ctrl_f)
        else:
            print(feature_line, file=train_feature_f)
            print(ctrl_line, file=train_ctrl_f)

    train_feature_f.close()
    train_ctrl_f.close()
    validate_feature_f.close()
    validate_ctrl_f.close()



if __name__ == "__main__":
    # prepare_wav_manifest(
    #     path_dirs="/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_audios",
    #     save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/wav.txt"
    # )

    # resample_audios(
    #     wav_manifest_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_wav.txt",
    #     manifest_save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_wav_16000.txt",
    #     audio_save_dir="/data/lipsync/xgyang/e2e_data/resampled_yingrao_dataproc_crop_audios",
    #     tgt_sample_rate=16000
    # )

    # prepare_both_manifest(
    #     audio_path_dirs="/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_audios",
    #     ctrl_path_dirs="/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05",
    #     audio_save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_wav.txt",
    #     ctrl_save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_ctrl.txt"
    # )

    # prepare_static_feature_manifest(
    #     path_dirs="/data/lipsync/xgyang/e2e_data/static_feature/hubert_50",
    #     save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_hubert_50.txt"
    # )

    # prepare_static_feature_manifest(
    #     path_dirs="/data/lipsync/xgyang/e2e_data/static_feature/wavlm_50",
    #     save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_wavlm_50.txt"
    # )

    # prepare_static_feature_manifest(
    #     path_dirs="/data/lipsync/xgyang/e2e_data/static_feature/wav2vec2_50",
    #     save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_wav2vec2_50.txt"
    # )

    # prepare_static_feature_both_manifest(
    #     feature_path_dirs="/data/lipsync/xgyang/e2e_data/static_feature/hubert_60",
    #     ctrl_path_dirs="/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_lumi05",
    #     feature_save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_huebrt_60.txt",
    #     ctrl_save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_ctrl.txt"
    # )


    split_validate_static_feature(
        feature_manifest_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_huebrt_60.txt", 
        ctrl_manifest_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_ctrl.txt", 
        save_dir="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir", 
        validate_p=30
    )



