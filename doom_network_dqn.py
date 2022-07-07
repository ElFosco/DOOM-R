import torch.nn as nn



class DoomDQN(nn.Module):
    def __init__(self, n_actions,device):
        '''
        Definition of the actor networks
        :param input_shape: size of the input that the network will receive
        :param n_actions: number of actions that the actor can make
        '''
        super(DoomDQN,self).__init__()
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
            nn.Linear(in_features= 1600, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.n_action)
        )

    def forward(self, input):
        '''
        :param x: input of the networks
        :return: probabilities of picking certain actions, state value
        '''
        input = input.to(self.device)
        return self.network(input)