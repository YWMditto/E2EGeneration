

from dataclasses import dataclass, field, fields, Field, MISSING, InitVar
import yaml
from pathlib import Path






def check_config_validity(*DataclassConfig):
    """
    检查不同 dataclass config 之间的 key 是否冲突；
    """

    all_keys = {}
    for dataclass_config in DataclassConfig:
        cur_keys = set(w.name for w in fields(dataclass_config))
        for past_name, past_keys in all_keys.items():
            if key := cur_keys.intersection(past_keys):
                raise RuntimeError(f"{dataclass_config.__name__} has key: {key} repetition with {past_name}.")
        all_keys[dataclass_config.__name__] = cur_keys


def parse_config_from_yaml(config, *DataclassConfig):

    """
    yaml 中如果两个字典的名字相同，那么后定义的字典会直接覆盖掉前面的字典，而并不会报错；

    TODO 加上类型检查，检查 yaml 中每一个值是否能够和 dataclass config 中设置的类型对应上；
    """
    if isinstance(config, (str, Path)):
        with open(config, "r") as f:
            config = yaml.load(f, yaml.FullLoader)

    all_keys = {}
    ins_config_dict = {}
    for dataclass_config in DataclassConfig:
        dataclass_name = dataclass_config.__name__
        cur_keys = set(w.name for w in fields(dataclass_config))
        for past_name, past_keys in all_keys.items():
            if key := cur_keys.intersection(past_keys):
                raise RuntimeError(f"{dataclass_name} has key: {key} repetition with {past_name}.")
        all_keys[dataclass_name] = cur_keys

        key_words = config.get(dataclass_name, {})
        if non_keys := set(key_words).difference(cur_keys):
            raise RuntimeError(f"{non_keys} of {dataclass_name} in yaml do not exist.")
        ins_config_dict[dataclass_name] = dataclass_config(**key_words)

    for config_name, config_dict in config.items():
        if config_name not in all_keys:
            raise RuntimeError(f"{config_name} does not exist.")

    return ins_config_dict




if __name__ == '__main__':




    with open("/remote-home/xgyang/E2EGeneration/V1/config/v1.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    a = 1





