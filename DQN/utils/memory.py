from collections import deque, namedtuple
import random
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class ReplayBuffer(object):
    def __init__(self, args):
        self.args = args
        self.buffer = deque(maxlen=args.buffer_max_size)  # First in first out
        self.Transition = namedtuple('transition', ('state', 'action', 'next_state', 'reward', 'done_mask'))

    def buffer_in(self, transition):
        # mapping values to a namedtuple(Transition)
        experience = self.Transition(**{k: v for k, v in zip(self.Transition.__dict__['_fields'], transition)})
        # add experience into the buffer
        self.buffer.append(experience)

    def buffer_out(self, n):

        # sampling experience from the buffer
        mini_batch = list(map(list, np.transpose(random.sample(self.buffer, n))))

        # allocate vectors to variables
        if self.args.input_type == 'linear':
            states = torch.tensor(mini_batch[0], dtype=torch.float).to(self.args.device)
            next_states = torch.tensor(mini_batch[2], dtype=torch.float).to(self.args.device)

        elif self.args.input_type == 'conv':
            states = torch.squeeze(torch.tensor(mini_batch[0], dtype=torch.float)).to(self.args.device)
            next_states = torch.squeeze(torch.tensor(mini_batch[2], dtype=torch.float)).to(self.args.device)

        actions = torch.tensor(list(map(lambda x: [x], mini_batch[1]))).to(self.args.device)
        rewards = torch.tensor(list(map(lambda x: [x], mini_batch[3]))).to(self.args.device)
        done_masks = torch.tensor(list(map(lambda x: [x], mini_batch[4]))).to(self.args.device)

        return states, actions, next_states, rewards, done_masks

    def size(self):
        return len(self.buffer)

