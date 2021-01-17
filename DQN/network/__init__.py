import os
import pandas as pd
from utils.args import DQN_parser, experiment_DQN_parser

# parser = DQN_parser()
# args = parser.parse_args()

config_df = pd.read_csv('./experiment_configs_conv.csv')
n = len(os.listdir('./experiment/'))
parser = experiment_DQN_parser(config_df, n)
args = parser.parse_args()

if args.input_type == 'linear':
    from network.dqn import Linear_Qnet as Qnet
elif args.input_type == 'conv':
    from network.dqn import Conv_Qnet as Qnet


