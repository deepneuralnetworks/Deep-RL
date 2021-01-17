from env.sim import env_gym
from network import Qnet
from utils.args import DQN_parser, experiment_DQN_parser
from utils.get_sim_image import RetrieveImageGym
from utils.memory import ReplayBuffer
from utils.mypath import path_summary, path_save, path_result
from tqdm import tqdm
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

import time

def train(args, q_online, q_target, memory, optimizer):

    for i in range(10):
        # get samples from the buffer
        states, actions, next_states, rewards, done_masks = memory.buffer_out(args.batch_size)

        # q-values from approximated q-function
        q_out = q_online(states)

        # get columns of action
        q_action = q_out.gather(1, actions)

        # select the action with the highest estimated q-value
        max_q_prime = q_target(next_states).max(1)[0].unsqueeze(1)

        # get target reward
        # no next state reward for terminal state(done_mask=0)
        target = rewards + (args.gamma * max_q_prime * done_masks)

        # get loss
        loss = F.smooth_l1_loss(q_action, target)

        # update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss


def main(args):
    env = env_gym(args.game)
    env.reset()

    # [conv] get rendering image from environment
    if args.input_type == 'linear':
        pass
    elif args.input_type == 'conv':
        transform_compose = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(40, interpolation=Image.CUBIC),
            transforms.ToTensor()
        ])
        img_getter = RetrieveImageGym(args, env, transform_compose)

    # get the size of action & state
    action_size = env.space_size(space='output')
    if args.input_type == 'linear':
        observation_size = env.space_size(space='input')
    elif args.input_type == 'conv':
        _, _, h, w = img_getter.get_screen().shape
        observation_size = (h, w)

    # define online network & target network
    q_online = Qnet(args, action_size=action_size, observation_size=observation_size).to(args.device)
    q_target = Qnet(args, action_size=action_size, observation_size=observation_size).to(args.device)
    q_target.load_state_dict(q_online.state_dict())

    # define replay buffer
    memory = ReplayBuffer(args)

    # define optimizer
    optimizer = optim.Adam(q_online.parameters(), lr=args.learning_rate)

    # define tensorboard writer
    writer = SummaryWriter(path_summary)

    # set utility options
    print_interval = args.print_interval

    # maximum episodes to perform
    max_episode = args.max_episode

    for n_epi in range(max_episode):

        # score & loss reset
        score = 0.0
        loss = 0.0

        # define & update epsilon rate
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))

        # reset gym to perform new episode
        if args.input_type == 'linear':
            state = env.reset()

        elif args.input_type == 'conv':
            env.reset()
            last_screen = img_getter.get_screen()
            current_screen = img_getter.get_screen()
            state = current_screen - last_screen

        # update the number of current iteration
        n_iter = 0

        # define done to identify whether the step is the end of episode
        done = False

        # start episode
        while not done:
            # select action by considering state
            if args.input_type == 'linear':
                action = q_online.sample_action(torch.from_numpy(state).float().to(args.device), epsilon)
            elif args.input_type == 'conv':
                action = q_online.sample_action(state.float().to(args.device), epsilon)

            # values from simulation
            next_state, reward, done, info = env.step(action)

            # done_mask get value 0.0 if the episode ends
            done_mask = 0.0 if done else 1.0

            # store buffer with experience / update current state
            if args.input_type == 'linear':
                # Linear uses next_state from simulator
                memory.buffer_in((state, action, next_state, reward / 100.0, done_mask))
                state = next_state

            elif args.input_type == 'conv':
                # Conv doesn't use next_state from simulator
                last_screen = current_screen
                current_screen = img_getter.get_screen()
                next_state = current_screen - last_screen
                memory.buffer_in((state.cpu().numpy(), action, next_state.cpu().numpy(), reward / 100.0, done_mask))
                state = next_state.to(args.device)

            # update n_iter per step
            n_iter += 1

            # accumulate rewards to monitor the process of training
            score += reward

            # break the loop if a episode ends
            if done:
                break

        # start training after enough experiences stored into the buffer
        if memory.size() > args.memory_threshold_to_start_train:
            loss = train(args, q_online, q_target, memory, optimizer)

        ## checking how is the training goes on
        # if n_epi % print_interval == 0 and n_epi != 0:
        #     print(f"n_episode :{n_epi}, score : {score:.1f}, n_buffer : {memory.size()}, eps : {epsilon * 100:.1f}%")

        # perform target network update
        if n_epi % args.target_update_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_online.state_dict())

        # tensorboard update
        writer.add_scalar(tag='score', scalar_value=score, global_step=n_epi)
        writer.add_scalar(tag=f'train_loss(after {args.memory_threshold_to_start_train} experience stored in memory)',
                          scalar_value=loss, global_step=n_epi)
        writer.add_scalar(tag='epsilon', scalar_value=epsilon, global_step=n_epi)

    # close simulation
    env.close()


if __name__ == '__main__':

    import pandas as pd
    config_df = pd.read_csv('./experiment_configs.csv')
    exp_lst = config_df.exp.tolist()

    n = len(os.listdir('./experiment/'))
    path_summary = f'./experiment/exp_{n}/summary/'

    for exp in exp_lst:
        print(f'Start Experiment-{exp:02d}')

        n = len(os.listdir('./experiment/'))
        path_summary = f'./experiment/exp_{n}/summary/'

        # Define parser
        parser = experiment_DQN_parser(config_df, exp)
        args = parser.parse_args()
        print(args)

        # start training
        main(args)




    # # Define parser
    # parser = DQN_parser()
    # args = parser.parse_args()
    #
    # # start training
    # main(args)
