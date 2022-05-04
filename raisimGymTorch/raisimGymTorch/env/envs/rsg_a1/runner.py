from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.rsg_a1 import RaisimGymEnv
from raisimGymTorch.env.bin.rsg_a1 import NormalSampler
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import argparse


if __name__ == '__main__':
    # task specification
    task_name = "anymal_locomotion"

    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default=None)
    args = parser.parse_args()
    weight_path = args.weight

    # check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # directories
    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = task_path + "/../../../../.."

    # config
    cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

    # create environment from the configuration file
    env = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)))
    env.reset()

    # shortcuts
    ob_dim = env.num_obs
    act_dim = env.num_acts
    num_threads = cfg['environment']['num_threads']

    # Training
    n_steps = 128
    total_steps = n_steps * env.num_envs
    avg_rewards = np.zeros(env.num_envs)
    avg_steps = np.zeros(env.num_envs)

    actor = ppo_module.Actor(
        ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
        ppo_module.MultivariateGaussianDiagonalCovariance(
            act_dim, env.num_envs, 1.0, NormalSampler(act_dim), cfg['seed']),
        device
    )

    critic = ppo_module.Critic(
        ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
        device
    )

    saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                               save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])

    # tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

    ppo = PPO.PPO(
        actor=actor,
        critic=critic,
        num_envs=cfg['environment']['num_envs'],
        num_transitions_per_env=n_steps,
        num_learning_epochs=4,
        gamma=0.996,
        lam=0.95,
        num_mini_batches=4,
        device=device,
        log_dir=saver.data_dir,
        shuffle_batch=True,
    )

    if weight_path is not None:
        load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

    for update in range(1000000):
        start = time.time()

        if update % cfg['environment']['eval_every_n'] == 0:
            print("Visualizing and evaluating the current policy")
            torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
            }, saver.data_dir+"/full_"+str(update)+'.pt')
            # we create another graph just to demonstrate the save/load method
            env.save_scaling(saver.data_dir, str(update))

        # actual training
        for step in range(n_steps):
            obs = env.observe()
            action = ppo.act(obs)
            reward, dones = env.step(action)
            ppo.step(value_obs=obs, rews=reward, dones=dones)
            avg_rewards = (avg_rewards + reward) * (1 - dones)
            avg_steps = (avg_steps + 1) * (1 - dones)

        # take st step to get value obs
        obs = env.observe()
        ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
        average_ll_performance = np.mean(avg_rewards)
        average_steps = np.mean(avg_steps)

        actor.update()
        # actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device))

        # curriculum update. Implement it in Environment.hpp
        if update % 30 == 0:
            env.curriculum_callback()

        end = time.time()

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("steps: ", '{:0.6f}'.format(average_steps)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
        print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                           * cfg['environment']['control_dt'])))
        print('----------------------------------------------------\n')
