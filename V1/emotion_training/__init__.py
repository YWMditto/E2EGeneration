
import torch


from .model import EmotionClassifier, EmotionClassifierConfig
from .train import UsedDatasetConfig, TrainingConfig
from helper_fns import parse_config_from_yaml



def load_emotion_classifier_ckpt(training_config_path, ckpt_path):
    config_dict = parse_config_from_yaml(training_config_path, UsedDatasetConfig, EmotionClassifierConfig, TrainingConfig)
    model_config: EmotionClassifierConfig = config_dict['EmotionClassifierConfig']
    model = EmotionClassifier(model_config)

    state_dict = torch.load(ckpt_path)
    state_dict = state_dict["state_dict"]
    # 去除 prefix；
    prefix = "model"
    len_prefix = len(prefix) + 1
    fixed_state_dict = {}
    for key, value in state_dict.items():
        fixed_state_dict[key[len_prefix:]] = value
    model.load_state_dict(fixed_state_dict)

    # 加载模型默认设置为 eval；
    model.eval()
    return model



