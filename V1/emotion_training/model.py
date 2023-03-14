



from dataclasses import dataclass

import torch
import torch.nn as nn




@dataclass
class EmotionClassifierConfig:
    emotion_num: int = 18  # 17 + pad(0)，因为我们数据预处理时是这样处理的；

    temp: float = 1.
    label_smoothing: float = 0.


class EmotionClassifier(nn.Module):
    def __init__(self, model_config: EmotionClassifierConfig):
        super(EmotionClassifier, self).__init__()

        emotion_num = model_config.emotion_num

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(input_size=256, hidden_size=384, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(in_features=384, out_features=384, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=384, out_features=384, bias=True),
            nn.Tanh(),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(in_features=384, out_features=384, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=384, out_features=emotion_num, bias=True),
        )

        self.cross_entropy_loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=model_config.label_smoothing)
        
        self.model_config = model_config

    def forward(self, batch):
        collated_ctrl_labels = batch["collated_ctrl_labels"]
        collated_ctrl_labels = collated_ctrl_labels.unsqueeze(1).permute(0, 1, 3, 2)

        hidden_state = self.conv(collated_ctrl_labels)  # [B, 1, 66, N']

        N, C, H, W = hidden_state.shape
        hidden_state = hidden_state.view(N, C*H, W)  # [B, 128*2, N']
        hidden_state = hidden_state.permute((0, 2, 1))  # [B, N', 256]

        hidden_state, _ = self.gru(hidden_state)  # [B, N', 384]
        hidden_state = self.fc(hidden_state[:, -1, :])  # [B, 384]

        return hidden_state
    
    def train_step(self, batch):
        hidden_state = self(batch)
        logits = self.projection_head(hidden_state)  # [B, 18]
        logits = logits / self.model_config.temp

        emotion_labels = batch["emotion_labels"]
        loss = self.cross_entropy_loss_fn(logits, emotion_labels)

        return {
            "loss": loss,
        }
    

    def evaluate_step(self, batch):
        hidden_state = self(batch)
        logits = self.projection_head(hidden_state)  # [B, 18]
        logits = logits / self.model_config.temp

        emotion_labels = batch["emotion_labels"]
        loss = self.cross_entropy_loss_fn(logits, emotion_labels)

        logits = nn.functional.softmax(logits, dim=-1)
        _, preds = torch.max(logits, dim=-1)
        return {"loss": loss, "preds": preds}






if __name__ == "__main__":

    a = torch.rand(2, 320, 66)
    model = EmotionClassifier(EmotionClassifierConfig())

    batch = {"collated_ctrl_labels": a}
    logits, hidden_state = model(batch)

    a = 1





