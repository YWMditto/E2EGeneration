



"""
修改音频和音素序列的长度，使得它们经过采样后或者直接与控制器标签的长度对齐；


"""

import sys
sys.path.append(".")
sys.path.append("..")


from pathlib import Path
import numpy as np
from scipy import interpolate
import soundfile as sf

import torch

from utils import _process_bar
from helper_fns import align_length_with_directly_insert


def read_ctrl_label_lengths(ctrl_label_dir, length_dict_save_path):

    length_dict_save_path = Path(length_dict_save_path)
    length_dict_save_path.parent.mkdir(exist_ok=True, parents=True)
    length_dict_f = open(length_dict_save_path, "w")

    ctrl_label_dir = Path(ctrl_label_dir)
    for path in ctrl_label_dir.iterdir():
        name = path.stem
        data = torch.load(path)
        length = len(data)
        print(f"{name}\t{length}", file=length_dict_f)

    length_dict_f.close()    



def read_ctrl_label_length_dict(ctrl_label_length_dict_path):
    length_dict = {}
    with open(ctrl_label_length_dict_path, "r") as f:
        for line in f:
            name, length = line.rstrip().split("\t")
            length_dict[name] = int(length)
    
    return length_dict


def align_audio(ctrl_label_length_dict, audio_dir, aligned_audio_save_dir):

    aligned_audio_save_dir = Path(aligned_audio_save_dir)
    aligned_audio_save_dir.mkdir(exist_ok=True, parents=True)

    audio_dir = Path(audio_dir)

    for path in audio_dir.iterdir():
        name = path.stem
        if name in ctrl_label_length_dict:
            length = ctrl_label_length_dict[name]
            length = round(length / 60 * 16000)

            audio, _ = sf.read(path)
            x = np.arange(0, len(audio))
            y = audio
            f = interpolate.interp1d(x, y, kind="linear")
            new_x = np.linspace(0, len(audio)-1, length)
            new_y = f(new_x)

            sf.write(aligned_audio_save_dir.joinpath(name+".wav"), new_y, 16000)
    


def align_static_feature(ctrl_label_length_dict, feature_dir, aligned_feature_save_dir):
    """
    TODO 这里先直接使用 resample_feature_from_50_to_60；
    """

    aligned_feature_save_dir = Path(aligned_feature_save_dir)
    aligned_feature_save_dir.mkdir(exist_ok=True, parents=True)

    feature_dir = Path(feature_dir)
    paths = list(feature_dir.iterdir())

    with _process_bar("align static feature", total=len(paths)) as update:
        for path in paths:
            name = path.stem
            if name in ctrl_label_length_dict:
                length = ctrl_label_length_dict[name]

                feature = torch.load(path)
                feature_length = len(feature)

                if feature_length > length:
                    aligned_feature = feature[:length]
                elif feature_length < length:
                    aligned_feature = align_length_with_directly_insert(feature, length)
                else:
                    aligned_feature = feature

                torch.save(aligned_feature, aligned_feature_save_dir.joinpath(name + ".pt"))

            update(1)


# pca 的长度和 ctrl label (lumi05) 的长度完全相同；
def check_pca_label(ctrl_label_length_dict, pca_dir):
    pca_dir = Path(pca_dir)

    pca_paths = list(pca_dir.iterdir())
    with _process_bar("check pca feature", total=len(pca_paths)) as update:
        for path in pca_paths:
            if path.stem in ctrl_label_length_dict:
                pca_data = torch.load(path)
                assert len(pca_data) == ctrl_label_length_dict[path.stem]
            update(1)




if __name__ == "__main__":
    # read_ctrl_label_lengths(
    #     ctrl_label_dir="/data/lipsync/xgyang/e2e_data/normalized_extracted_ctrl_labels",
    #     length_dict_save_path="/data/lipsync/xgyang/e2e_data/ctrl_label_length_dict.txt"
    # )


    length_dict = read_ctrl_label_length_dict("/data/lipsync/xgyang/e2e_data/ctrl_label_length_dict.txt")

    # align_audio(
    #     ctrl_label_length_dict=length_dict,
    #     audio_dir="/data/lipsync/xgyang/e2e_data/resampled_yingrao_dataproc_crop_audios",
    #     aligned_audio_save_dir="/data/lipsync/xgyang/e2e_data/aligned_audios"
    # )

    # align_static_feature(
    #     ctrl_label_length_dict=length_dict,
    #     feature_dir="/data/lipsync/xgyang/e2e_data/static_feature/layer6/hubert_50",
    #     aligned_feature_save_dir="/data/lipsync/xgyang/e2e_data/static_feature/layer6/hubert_60"
    # )

    # align_static_feature(
    #     ctrl_label_length_dict=length_dict,
    #     feature_dir="/data/lipsync/xgyang/e2e_data/static_feature/layer6/wavlm_50",
    #     aligned_feature_save_dir="/data/lipsync/xgyang/e2e_data/static_feature/layer6/wavlm_60"
    # )

    # align_static_feature(
    #     ctrl_label_length_dict=length_dict,
    #     feature_dir="/data/lipsync/xgyang/e2e_data/static_feature/layer6/wav2vec2_50",
    #     aligned_feature_save_dir="/data/lipsync/xgyang/e2e_data/static_feature/layer6/wav2vec2_60"
    # )

    # align_static_feature(
    #     ctrl_label_length_dict=length_dict,
    #     feature_dir="/data/lipsync/xgyang/e2e_data/static_feature/ser_hubert/50/layer24",
    #     aligned_feature_save_dir="/data/lipsync/xgyang/e2e_data/static_feature/ser_hubert/60/layer24"
    # )

    check_pca_label(
        ctrl_label_length_dict=length_dict,
        pca_dir="/data/lipsync/xgyang/e2e_data/yingrao/dataproc/crop_pca"
    )





