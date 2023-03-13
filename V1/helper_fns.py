
import logging


from dataclasses import dataclass, field, fields, Field, MISSING, InitVar, is_dataclass
import yaml
from pathlib import Path

import torch
import torch.nn as nn

import numpy as np


logger = logging.getLogger(__file__)



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
    ins_config_dict = {}
    if config is None:
        logger.warning("yaml config file is None, notice whether this is what you want.")
        for dataclass_config in DataclassConfig:
            ins_config_dict[dataclass_config.__name__] = dataclass_config()
        return ins_config_dict

    if isinstance(config, (str, Path)):
        with open(config, "r") as f:
            config = yaml.load(f, yaml.FullLoader)

    all_keys = {}
    for dataclass_config in DataclassConfig:
        dataclass_name = dataclass_config.__name__
        cur_keys = set()
        sub_dataclasses = {}
        for _field in fields(dataclass_config):
            _field_name = _field.name
            cur_keys.add(_field_name)
            if is_dataclass(_field.type):
                sub_dataclasses[_field_name] = _field.type
        for past_name, past_keys in all_keys.items():
            if key := cur_keys.intersection(past_keys):
                raise RuntimeError(f"{dataclass_name} has key: {key} repetition with {past_name}.")
        all_keys[dataclass_name] = cur_keys

        key_words = config.get(dataclass_name, {})
        if non_keys := set(key_words).difference(cur_keys):
            raise RuntimeError(f"{non_keys} of {dataclass_name} in yaml do not exist.")
        for _sub_dataclass_key, _sub_dataclass_value in sub_dataclasses.items():
            if _sub_dataclass_key in key_words:
                key_words[_sub_dataclass_key] = parse_config_from_yaml(dict([(_sub_dataclass_value.__name__, key_words[_sub_dataclass_key])]), _sub_dataclass_value)[_sub_dataclass_value.__name__]
        ins_config_dict[dataclass_name] = dataclass_config(**key_words)

    for config_name, config_dict in config.items():
        if config_name not in all_keys:
            raise RuntimeError(f"{config_name} does not exist.")

    return ins_config_dict



def _init_weights(module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight.data)
    if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
        module.bias.data.zero_()



def align_length_with_directly_insert(data, tgt_length):

    length = len(data)
    if length == tgt_length:
        return data
    assert length < tgt_length

    insert_num = tgt_length - length
    gap_num = length // insert_num

    
    if isinstance(data, torch.Tensor):
        new_data = data.new_zeros(size=(tgt_length, ) + data.shape[1:])
    elif isinstance(data, np.ndarray):
        new_data = np.zeros((tgt_length, ) + data.shape[1:])
    else:
        new_data = [None for _ in range(tgt_length)]

    for i in range(insert_num):
        new_data[i*gap_num+i:(i+1)*gap_num+i] = data[i*gap_num:(i+1)*gap_num]
        new_data[(i+1)*gap_num+i] = new_data[max((i+1)*gap_num+i - 1, 0)]

    new_data[(i+1)*gap_num+i+1:] = data[(i+1)*gap_num:]
    return new_data


if __name__ == '__main__':




    with open("/remote-home/xgyang/E2EGeneration/V1/config/v1.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    a = 1





