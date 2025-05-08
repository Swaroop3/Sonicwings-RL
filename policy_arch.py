# policy_arch.py
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            dummy = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        return self.linear(self.cnn(obs / 255.0))

class CustomPolicy(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.features_extractor = CustomCNN(observation_space)
        self.action_net = nn.Linear(512, action_space.n)
        self.value_net = nn.Linear(512, 1)

    def forward(self, obs):
        features = self.features_extractor(obs)
        return self.action_net(features), self.value_net(features)
