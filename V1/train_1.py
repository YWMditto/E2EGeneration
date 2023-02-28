

"""
TODO 实现对于 Model1 的训练，之后应当从中抽取出共用的 train_utils；


"""



from dataclasses import dataclass




from data_prepare import (
    StaticFeatureDatasetConfig,
    StaticFeatureDataset,
    StaticFeatureCollater
)

from models import (
    Model1Config,
    Model1
)



@dataclass
class DDPConfig:
    find_unused_parameters: bool = False




@dataclass
class TrainingConfig:
    one_batch_total_tokens: int = 6000
    shuffle: bool = True

    
    base_lr: float = 1e-4
    encoder_lr: float = 2e-5  # 用户在配置 encoder optim params 的时候应该使用 pipeline config；
    mouth_head_lr: float = 4e-4
    eye_head_lr: float = 4e-4

    warmup_epochs: int = 2
    n_epochs: int = 20

    evaluate_every: int = -4  # TODO


    gradient_accumulation_step: int = 1
    ddp_config: DDPConfig = DDPConfig()



def train():


    """
    1. parse config;
    2. optimizer and lr schedulers;
    3. callbacks, maybe later;
    4. train loop;
    """
















if __name__ == "__main__":


    train()



































