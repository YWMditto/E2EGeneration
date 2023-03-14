

"""

为添加 emotion embedding 做准备；


暂时先只考虑如何在 model_1 中添加；

TODO 实现在 model_2 中 emotion embedding 的添加；

"""


from pathlib import Path
from collections import defaultdict


import torch


from utils import _process_bar



def stat_emotion_class(ctrl_label_dir):
    """
    17
    {'neutral': 3658, 'happy-little': 187, 'happy-medium': 194, 'sad-little': 209, 'sad-medium': 205, 'cute': 244, 'surprise-great': 198, 'surprise-medium': 190, 'afraid-great': 160, 
    'sad-great': 209, 'afraid-little': 158, 'happy-great': 192, 'angry-medium': 95, 'surprise-little': 177, 'angry-little': 78, 'afraid-medium': 160, 'angry-great': 96}                                                                                                                           
    """

    ctrl_label_dir = Path(ctrl_label_dir)
    paths = list(ctrl_label_dir.iterdir())

    emotion_dict = defaultdict(lambda: 0)

    with _process_bar("stat emotion class", total=len(paths)) as update:
        for path in paths:
            name = path.stem

            emotion = name.split("-")
            emotion = "-".join(emotion[1:])
            emotion_dict[emotion] += 1

            update(1)

    emotion_dict = dict(emotion_dict)

    return emotion_dict


EMOTION_DICT = {
    'pad': 0, 'neutral': 1, 'happy-little': 2, 'happy-medium': 3, 'sad-little': 4, 'sad-medium': 5, 'cute': 6, 'surprise-great': 7, 'surprise-medium': 8, 'afraid-great': 9, 
    'sad-great': 10, 'afraid-little': 11, 'happy-great': 12, 'angry-medium': 13, 'surprise-little': 14, 'angry-little': 15, 'afraid-medium': 16, 'angry-great': 17
}

REMAP_EMOTION_DICT = {0: 'pad', 1: 'neutral', 2: 'happy-little', 3: 'happy-medium', 4: 'sad-little', 5: 'sad-medium', 6: 'cute', 7: 'surprise-great', 8: 'surprise-medium', 
    9: 'afraid-great', 10: 'sad-great', 11: 'afraid-little', 12: 'happy-great', 13: 'angry-medium', 14: 'surprise-little', 15: 'angry-little', 16: 'afraid-medium', 17: 'angry-great'
}



def generate_emotion_feature(ctrl_label_dir, save_dir):
    ctrl_label_dir = Path(ctrl_label_dir)
    paths = list(ctrl_label_dir.iterdir())

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    with _process_bar("stat emotion class", total=len(paths)) as update:
        for path in paths:
            name = path.stem
            emotion = name.split("-")
            emotion = "-".join(emotion[1:])

            emotion_idx = EMOTION_DICT[emotion]
            torch.save(emotion_idx, save_dir.joinpath(name + ".pt"))

            update(1)




if __name__ == "__main__":

    # emotion_dict = stat_emotion_class(
    #     "/data/lipsync/xgyang/e2e_data/normalized_extracted_ctrl_labels"
    # )

    # print(emotion_dict)
    # print(len(emotion_dict))


    generate_emotion_feature(
        ctrl_label_dir="/data/lipsync/xgyang/e2e_data/normalized_extracted_ctrl_labels",
        save_dir="/data/lipsync/xgyang/e2e_data/emotions"
    )









