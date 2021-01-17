import torch.nn as nn
import torch.nn.functional as F

import random


class Linear_Qnet(nn.Module):
    def __init__(self, args, action_size, observation_size):
        super(Linear_Qnet, self).__init__()
        self.args = args
        self.action_size = action_size
        self.observation_size = observation_size
        self.fc1 = nn.Linear(observation_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, self.action_size-1)
        else:
            return out.argmax().item()



class Conv_Qnet(nn.Module):
    def __init__(self, args, action_size, observation_size):
        super(Conv_Qnet, self).__init__()
        self.args = args
        self.action_size = action_size
        w, h = observation_size

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so comput it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1)//stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, action_size)

    def forward(self, x):

        # Conv encoders
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # make linear feature vector
        x = self.head(x.view(x.size(0), -1))
        
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, self.action_size-1)
        else:
            return out.argmax().item()