

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


def prepare_other_feature_manifest(main_fest_path, feature_dirs, save_path):
    if not isinstance(feature_dirs, (str, Path)):
        feature_dirs = [feature_dirs]

    search_model = SearchFilePath()
    all_path_list = search_model.find_all_file_paths(
        data_dirs=feature_dirs,
        pattern=".+\.pt",
        n_proc=64,
        depth=1
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    save_f = open(save_path, "w")

    main_fest_names = []
    with open(main_fest_path, "r") as f:
        for line in f:
            main_path, _ = line.rstrip().split("\t")
            main_path = Path(main_path)
            main_fest_names.append(main_path.stem)

    feature_fest_names = {}
    for _path in all_path_list:
        _path = Path(_path)
        feature_fest_names[_path.stem] = str(_path)

    with _process_bar("prepare other feature manifest", total=len(main_fest_names)) as update:
        for _name in main_fest_names:
            try:
                _path = feature_fest_names[_name]      
                print(_path, file=save_f)
            except:
                print(_name)

            update(1)



def split_validate_many_feature(feature_tuples, validate_p=30):

    def read_txt(file_path):
        with open(file_path, 'r') as f:
            data_list = [line.rstrip() for line in f]
        return data_list

    feature_list_list = []
    length = None
    for i in range(len(feature_tuples)):
        feature_list = read_txt(feature_tuples[i])
        feature_list_list.append(feature_list)

        if length is None:
            length = len(feature_list)
        else:
            assert length == len(feature_list)

    if isinstance(validate_p, float):
        assert 0 < validate_p < 1
        validate_p = int(length * validate_p)
    
    validate_indices = random.sample(range(len(feature_list)), validate_p)

    train_feature_f_list, validate_feature_f_list = [], []
    for i in range(len(feature_tuples)):
        feature_path = Path(feature_tuples[i])
        train_feature_mainfest_path = feature_path.parent.joinpath(feature_path.stem + "_train" + feature_path.suffix)
        validate_feature_mainfest_path = feature_path.parent.joinpath(feature_path.stem + "_validate" + feature_path.suffix)

        train_feature_f = open(train_feature_mainfest_path, "w")
        validate_feature_f = open(validate_feature_mainfest_path, "w")

        train_feature_f_list.append(train_feature_f)
        validate_feature_f_list.append(validate_feature_f)

    validate_indices = set(validate_indices)
    for idx, lines in enumerate(zip(*feature_list_list)):
        for i in range(len(feature_list_list)):
            if idx in validate_indices:
                print(lines[i], file=validate_feature_f_list[i])
            else:
                print(lines[i], file=train_feature_f_list[i])

    for each_f in train_feature_f_list + validate_feature_f_list:
        each_f.close()



def prepare_many_manifest(feature_tuples):
    """
    feature_tuples: 
        (feature_dirs, save_path, pattern, load_length: bool)
    """
    
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

    
    def _conv_name_dict(path_list) -> dict:
        tmp_dict = {}
        for _path in path_list:
            _path = Path(_path)
            tmp_dict[_path.stem] = str(_path)
        return tmp_dict


    feature_f_list, feature_dict_list = [], []
    used_feature_set = None
    load_lengths = []
    for _tuple in feature_tuples:
        feature_path_list, feature_f = _prepare(_tuple[0], _tuple[1], _tuple[2], 1)
        feature_f_list.append(feature_f)

        feature_dict = _conv_name_dict(feature_path_list)
        feature_dict_list.append(feature_dict)
        feature_set = set(feature_dict)

        if used_feature_set is None:
            used_feature_set = feature_set
        else:
            used_feature_set &= feature_set

        load_lengths.append(_tuple[-1])
    

    with _process_bar("write manifest", total=len(used_feature_set)) as update:
        for _name in used_feature_set:
            
            for i in range(len(feature_dict_list)):
                feature_dict = feature_dict_list[i]
                _feature_path = feature_dict[_name]
                feature_line = str(_feature_path)
                if load_lengths[i]:
                    _feature = torch.load(_feature_path)
                    feature_length = len(_feature)
                    feature_line += "\t" + str(feature_length)

                feature_f = feature_f_list[i]
                print(feature_line, file=feature_f)

            update(1)

    for f in feature_f_list:
        f.close()


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
    #     ctrl_path_dirs="/data/lipsync/xgyang/e2e_data/normalized_extracted_ctrl_labels",
    #     feature_save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_huebrt_60.txt",
    #     ctrl_save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_normalized_extracted_ctrl.txt"
    # )

    # split_validate_static_feature(
    #     feature_manifest_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_huebrt_60.txt", 
    #     ctrl_manifest_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_normalized_extracted_ctrl.txt", 
    #     save_dir="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir", 
    #     validate_p=30
    # )


    # prepare_other_feature_manifest(
    #     main_fest_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_huebrt_60_train.txt",
    #     feature_dirs="/data/lipsync/xgyang/e2e_data/phoneme",
    #     save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_phn_train.txt"
    # )

    # prepare_other_feature_manifest(
    #     main_fest_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_feature_huebrt_60_validate.txt",
    #     feature_dirs="/data/lipsync/xgyang/e2e_data/phoneme",
    #     save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi05_phn_validate.txt"
    # )

    prepare_many_manifest(
        [
            (
                "/data/lipsync/xgyang/e2e_data/static_feature/hubert_60",
                "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_hubert_60.txt",
                ".+\.pt",
                True
            ),
            (
                "/data/lipsync/xgyang/e2e_data/normalized_extracted_ctrl_labels",
                "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_normalized_extracted_ctrl.txt",
                ".+\.pt",
                True
            ),
            (
                "/data/lipsync/xgyang/e2e_data/phoneme",
                "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_phn_60.txt",
                ".+\.pt",
                False
            )
        ]
    )

    split_validate_many_feature(
        [
            "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_hubert_60.txt",
            "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_normalized_extracted_ctrl.txt",
            "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_phn_60.txt"
        ],
        validate_p=30
    )
