#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_a1
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import time
import torch
import argparse

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
parser.add_argument('--viz', action='store_true')
args = parser.parse_args()

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
log_path = "/".join(args.weight.split("/")[:-1])
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 1
cfg['environment']['k_0'] = 1.0

env = VecEnv(
    rsg_a1.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
    normalize_ob=True
)
env.reset()

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path, map_location=device)['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    if args.viz:
        env.turn_on_visualization()

    for episode in range(100):
        done = False
        reward_ll_sum = 0
        steps = 0
        while not done:
            steps += 1
            if args.viz:
                time.sleep(0.01)
            obs = env.observe(False)
            action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
            reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
            reward_ll_sum += reward_ll[0]
            done = dones[0]
        print('----------------------------------------------------')
        print(steps)
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum)))
        print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format(steps * 0.01)))
        print('----------------------------------------------------\n')

    if args.viz:
        env.turn_off_visualization()
