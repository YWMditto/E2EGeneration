

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



def filter_length_from_manifest(manifest_with_length_paths, other_manifest_paths, save_dir, length_diff_thres: int = 1):
    
    if manifest_with_length_paths is None or len(manifest_with_length_paths) <= 1:
        return 

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)


    with_length_paths = []
    total_length = None
    for i in range(len(manifest_with_length_paths)):
        cur_with_length_manifest_path = manifest_with_length_paths[i]
        with open(cur_with_length_manifest_path, "r") as f:
            cur_paths = []
            for line in f:
                _path, _length = line.rstrip().split("\t")
                _length = int(_length)
                cur_paths.append((_path, _length))
        with_length_paths.append(cur_paths)

        if total_length is None:
            total_length = len(cur_paths)
        else:
            assert len(cur_paths) == total_length

    other_paths = []
    for i in range(len(other_manifest_paths)):
        cur_other_manifest_path = other_manifest_paths[i]
        with open(cur_other_manifest_path, "r") as f:
            cur_paths = []
            for line in f:
                line = line.rstrip()
                cur_paths.append(line)
        other_paths.append(cur_paths)

        assert len(cur_paths) == total_length


    with_length_save_f_list = []    
    for i in range(len(manifest_with_length_paths)):
        cur_with_length_manifest_path = Path(manifest_with_length_paths[i])
        cur_save_f = save_dir.joinpath(cur_with_length_manifest_path.name)
        cur_save_f = open(cur_save_f, "w")
        with_length_save_f_list.append(cur_save_f)
    
    other_save_f_list = []
    for i in range(len(other_manifest_paths)):
        cur_other_manifest_path = Path(other_manifest_paths[i])
        cur_save_f = save_dir.joinpath(cur_other_manifest_path.name)
        cur_save_f = open(cur_save_f, "w")
        other_save_f_list.append(cur_save_f)


    for i in range(total_length):
        should_save = True
        cur_length = None
        for j in range(len(with_length_paths)):
            cur_with_length_path, length = with_length_paths[j][i]
            if cur_length is None:
                cur_length = length
            else:
                if abs(cur_length - length) > length_diff_thres:
                    should_save = False
        
        if should_save:
            for j in range(len(with_length_paths)):
                line = with_length_paths[j][i][0] + "\t" + str(with_length_paths[j][i][1])
                print(line, file=with_length_save_f_list[j])
            
            for j in range(len(other_paths)):
                line = other_paths[j][i]
                print(line, file=other_save_f_list[j])

    for f in with_length_save_f_list + other_save_f_list:
        f.close()
            


def expand_manifest(filepath, save_path, repeat_times=2):
    save_path = Path(save_path)
    data_f = open(save_path, "w")
    with open(filepath) as f:
        data = []
        for line in f:
            data.append(line.rstrip())

    for i in range(repeat_times):
        for line in data:
            print(line, file=data_f)

    data_f.close()









if __name__ == "__main__":
    # prepare_wav_manifest(
    #     path_dirs="/data/lipsync/xgyang/e2e_data/aligned_audios",
    #     save_path="/data/lipsync/xgyang/e2e_data/static_feature/aligned_audios_manifest.txt"
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

    # prepare_many_manifest(
    #     [
    #         (
    #             "/data/lipsync/xgyang/e2e_data/static_feature/hubert_60",
    #             "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_hubert_60.txt",
    #             ".+\.pt",
    #             True
    #         ),
    #         (
    #             "/data/lipsync/xgyang/e2e_data/normalized_extracted_ctrl_labels",
    #             "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_normalized_extracted_ctrl.txt",
    #             ".+\.pt",
    #             True
    #         ),
    #         (
    #             "/data/lipsync/xgyang/e2e_data/phoneme",
    #             "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_phn_60.txt",
    #             ".+\.pt",
    #             False
    #         )
    #     ]
    # )

    # split_validate_many_feature(
    #     [
    #         "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_hubert_60.txt",
    #         "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_normalized_extracted_ctrl.txt",
    #         "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_phn_60.txt"
    #     ],
    #     validate_p=30
    # )


    # length_diff_thres = 1 
    # filter_length_from_manifest(
    #     manifest_with_length_paths=[
    #         "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_hubert_60_train.txt",
    #         "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_normalized_extracted_ctrl_train.txt",
    #     ],
    #     other_manifest_paths=[
    #         "/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/lumi_phn_60_train.txt"
    #     ],
    #     save_dir=f"/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/after_filter_{length_diff_thres}",
    #     length_diff_thres=length_diff_thres
    # )



    # expand_manifest(
    #     filepath="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/after_filter_1/lumi_hubert_60_train.txt",
    #     save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/after_filter_1/lumi_hubert_60_train_repeat.txt",
    #     repeat_times=20
    # )

    # expand_manifest(
    #     filepath="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/after_filter_1/lumi_normalized_extracted_ctrl_train.txt",
    #     save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/after_filter_1/lumi_normalized_extracted_ctrl_train_repeat.txt",
    #     repeat_times=20
    # )

    # expand_manifest(
    #     filepath="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/after_filter_1/lumi_phn_60_train.txt",
    #     save_path="/data/lipsync/xgyang/E2EGeneration/V1/cache_dir/after_filter_1/lumi_phn_60_train_repeat.txt",
    #     repeat_times=20
    # )



