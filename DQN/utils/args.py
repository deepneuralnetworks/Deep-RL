import argparse
import torch

def DQN_parser():
    parser = argparse.ArgumentParser(description='default: Vanila DQN w/ (Linear | Conv)')

    # environment option (gym)
    parser.add_argument('--game', type=str, default='CartPole-v0')

    # DQN option
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--input_type', type=str, default='linear',
                        help='linear or conv layer is available')

    ## Training option
    parser.add_argument('--max_episode', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.98,
                        help='decaying future reward')
    parser.add_argument('--memory_threshold_to_start_train', type=int, default=2000)
    parser.add_argument('--target_update_interval', type=int, default=20)

    # Memory option
    parser.add_argument('--buffer_max_size', type=int, default=50000)

    # etc option
    parser.add_argument('--print_interval', type=int, default=20)
    parser.add_argument('--device', type=int, default=torch.device('cpu'))

    return parser

def experiment_DQN_parser(config_df, exp):

    config = config_df.set_index('exp').loc[exp].to_dict()
    parser = argparse.ArgumentParser(description='default: Vanila DQN w/ (Linear | Conv)')

    # environment option (gym)
    parser.add_argument('--game', type=str, default='CartPole-v0')

    # DQN option
    parser.add_argument('--hidden_size', type=int, default=int(config['hidden_size']))
    parser.add_argument('--input_type', type=str, default=config['input_type'],
                        help='linear or conv layer is available')

    ## Training option
    parser.add_argument('--max_episode', type=int, default=int(config['max_episode']))
    parser.add_argument('--batch_size', type=int, default=int(config['batch_size']))
    parser.add_argument('--learning_rate', type=float, default=config['learning_rate'])
    parser.add_argument('--gamma', type=float, default=config['gamma'],
                        help='decaying future reward')
    parser.add_argument('--memory_threshold_to_start_train', type=int, default=int(config['memory_threshold_to_start_train']))
    parser.add_argument('--target_update_interval', type=int, default=int(config['target_update_interval']))

    # Memory option
    parser.add_argument('--buffer_max_size', type=int, default=int(config['buffer_max_size']))

    # etc option
    parser.add_argument('--print_interval', type=int, default=20)
    parser.add_argument('--device', type=str, default=torch.device('cuda:0' if config['input_type'] == 'conv' else 'cpu'))

    return parser

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('../experiment_configs_conv.csv')
    # print(df['device'])
    # print(df[df.exp == 0].to_dict())
    dd = df.set_index('exp').loc[12].to_dict()
    print(dd['device'])
    print(type(dd['device']))

    parser = experiment_DQN_parser(df, 12)
    args = parser.parse_args()
    print(args)