import argparse
import os

from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe
import torch.distributed as dist
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import copy


def interact(parent_conns, actions):
    ss = []
    rs = []
    ds = []
    rds = []
    lrs = []

    for parent_conn, action in zip(parent_conns, actions):
        parent_conn.send(action)

    for parent_conn in parent_conns:
        s, r, d, rd, lr = parent_conn.recv()
        ss.append(s)
        rs.append(r)
        ds.append(d)
        rds.append(rd)
        lrs.append(lr)

    return ss, rs, ds, rds, lrs


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def single_gpu_work(gpu, args):
    rank = args.node_rank * args.gpus + gpu
    dist.init_process_group(backend="nccl",
                            init_method='env://',
                            world_size=args.world_size,
                            rank=rank)

    torch.cuda.set_device(gpu)

    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    env_id = default_config['EnvID']
    EnvType = default_config['EnvType']

    if EnvType == 'mario':
        env = JoypadSpace(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif EnvType == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_load_model = False
    is_render = False
    model_path = 'models/{}.model'.format(env_id)
    icm_path = 'models/{}.icm'.format(env_id)

    if args.node_rank == 0:
        writer = SummaryWriter()

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_env = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_env / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    eta = float(default_config['ETA'])

    clip_grad_norm = float(default_config['ClipGradNorm'])

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 4, 84, 84))

    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(gamma)

    agent = ICMAgent

    if default_config['EnvType'] == 'atari':
        EnvType = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        EnvType = MarioEnvironment
    else:
        raise NotImplementedError

    agent = agent(
        input_size,
        output_size,
        num_env,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        eta=eta,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net,
        gpu=gpu
    )

    if is_load_model:
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
        else:
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))

    print("Opening environment processes and pipes")
    envs = []
    parent_conns = []
    child_conns = []
    for idx in range(num_env):
        parent_conn, child_conn = Pipe()
        env = EnvType(env_id, is_render, idx, child_conn)
        env.start()
        envs.append(env)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_env, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    # normalize obs
    print('Start to initailize observation normalization parameter.....')
    next_obs = []
    steps = 0
    while steps < pre_obs_norm_step:
        steps += num_env
        actions = np.random.randint(0, output_size, size=(num_env,))

        ss, rs, ds, rds, lrs = interact(parent_conns, actions)
        next_obs += ss

        # for parent_conn, action in zip(parent_conns, actions):
        #     parent_conn.send(action)
        #
        # for parent_conn in parent_conns:
        #     s, r, d, rd, lr = parent_conn.recv()
        #     next_obs.append(s[:])

    # TODO all reduce here to collect average
    next_obs = np.stack(next_obs)
    obs_rms.update(next_obs)
    print('End to initalize...')

    while True:
        total_state, total_reward, total_done, total_next_state, total_action, \
        total_int_reward, total_next_obs, total_values, total_policy = \
            [], [], [], [], [], [], [], [], []
        global_step += (num_env * num_step)
        global_update += 1

        # Step 1. n-step rollout
        for _ in range(num_step):
            # everything is detached.
            # the agent interacts in eval(), completely not differentiable.
            # TODO this is not a map reduce, because all gpus have the same weights
            actions, value, policy = agent.get_action(obs_rms.normalize(states))

            next_states, rewards, dones, real_dones, log_rewards = interact(parent_conns, actions)

            # for parent_conn, action in zip(parent_conns, actions):
            #     parent_conn.send(action)
            #
            # next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
            # for parent_conn in parent_conns:
            #     s, r, d, rd, lr = parent_conn.recv()
            #     next_states.append(s)
            #     rewards.append(r)
            #     dones.append(d)
            #     real_dones.append(rd)
            #     log_rewards.append(lr)

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)

            # total reward = int reward
            # TODO review rewards calculation
            intrinsic_reward = agent.compute_intrinsic_reward(
                obs_rms.normalize(states),
                obs_rms.normalize(next_states),
                actions)
            sample_i_rall += intrinsic_reward[sample_env_idx]

            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_next_state.append(next_states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_values.append(value)
            total_policy.append(policy)

            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_idx]

            sample_step += 1
            if real_dones[sample_env_idx] and args.node_rank==0:
                sample_episode += 1
                writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                writer.add_scalar('data/step', sample_step, sample_episode)
                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0

        # calculate last next value
        _, value, _ = agent.get_action(obs_rms.normalize(states))

        total_values.append(value)

        # --------------------------------------------------

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_values = np.stack(total_values).transpose()
        total_logging_policy = torch.stack(total_policy).view(-1, output_size).cpu().numpy()

        # Step 2. reform intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        if args.node_rank == 0:
            writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_env, sample_episode)
            writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_env, global_update)
            # -------------------------------------------------------------------------------------------

            # logging Max action probability
            writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        # Step 3. make target and advantage
        # TODO how is advantage calculated particularly?
        target, adv = make_train_data(total_int_reward,
                                      np.zeros_like(total_int_reward),
                                      total_values,
                                      gamma,
                                      num_step,
                                      num_env)

        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        # -----------------------------------------------

        # Step 4. Training!
        # TODO why do you take the next state as the input?
        agent.train_model(obs_rms.normalize(total_state),
                          obs_rms.normalize(total_next_state),
                          target, total_action,
                          adv,
                          total_policy)
        if args.node_rank == 0:
            if global_step % (num_env * num_step * 100) == 0:
                print('Now Global Step :{}'.format(global_step))
                torch.save(agent.model.state_dict(), model_path)
                torch.save(agent.icm.state_dict(), icm_path)


def main():
    mp.set_start_method('fork')
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--node_rank', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes  #
    os.environ['MASTER_ADDR'] = 'frost-6.las.iastate.edu'  #
    os.environ['MASTER_PORT'] = '8888'  #
    # you must fork, so that environments can be forked again. Do not spawn
    mp.start_processes(single_gpu_work, nprocs=args.gpus, args=(args,), start_method="fork")  #
    # single_gpu_work(0,args)  #
    #########################################################


if __name__ == '__main__':
    main()
