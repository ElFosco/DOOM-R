import torch
import torch.nn as nn



class DoomDDQN(nn.Module):
    def __init__(self, n_actions,device):
        super(DoomDDQN,self).__init__()
        self.n_action = n_actions
        self.device = device
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.Flatten(start_dim=0),
        )
        self.v = nn.Sequential(
            nn.Linear(in_features=1600, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )
        self.a = nn.Sequential(
            nn.Linear(in_features=1600, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.n_action)
        )

    def forward(self, input):
        input = input.to(self.device)
        output = self.network(input)
        v = self.v(output)
        a = self.a(output)
        output = v + a - torch.mean(a)
        return output