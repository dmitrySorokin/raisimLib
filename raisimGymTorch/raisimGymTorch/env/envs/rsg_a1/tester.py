from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_a1
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import time
import torch
import argparse
import numpy as np
from tqdm import trange
from pprint import pprint


if __name__ == '__main__':
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
        update_stats=False
    )
    obs = env.reset()

    # shortcuts
    ob_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

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

        episode_rewards = []
        episode_steps = []
        for episode in trange(10):
            done, total_reward, steps, info = False, 0, 0, {}
            while not done:
                steps += 1
                if args.viz:
                    time.sleep(0.01)
                action = loaded_graph.architecture(torch.from_numpy(obs))
                obs, rewards, dones, infos = env.step(action.detach().numpy())
                total_reward += rewards[0]
                done = dones[0]
                info = infos[0]
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            print('----------------------------------------------------')
            pprint(info)
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(total_reward)))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format(steps * 0.01)))
            print('----------------------------------------------------\n')
        print(f'return {np.mean(episode_rewards)} +- {np.std(episode_rewards)}')
        print(f'steps {np.mean(episode_steps)} +- {np.std(episode_steps)}')

        if args.viz:
            env.turn_off_visualization()
