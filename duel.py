"""Uses the DDDQN algorithm to train a neural model."""
import argparse
import gym
from gym.spaces import Box, Discrete
import numpy as np
from scipy.ndimage.interpolation import shift
import os
import time
from duel_aux import bcolors, print_results, ReplayHolder
from duel_model import load, DuelingModel
from exploration import get_strategy

parser = argparse.ArgumentParser()

# ========== TRAINING PARAMETERS
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--batch_norm', action="store_true", default=False)
parser.add_argument('--no_batch_norm', action="store_false", dest='batch_norm')
parser.add_argument('--training_start_size', type=int, default=5000)
parser.add_argument('--train_repeat', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--episodes', type=int, default=20000)
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
parser.add_argument('--update_frequency', type=int, default=4)
parser.add_argument('--target_net_update_frequency', type=int, default=32)
parser.add_argument('--replay_size', type=int, default=100000)
parser.add_argument('--max_timesteps', type=int, default=1500)

parser.add_argument('--prioritize', action="store_true", default=False)
parser.add_argument('--update_tree_interval', type=int, default=100000)

# parser.add_argument('--exploration_params', type=float, default=.1)
parser.add_argument('--exploration_strategy', choices=['semi_uniform', 'e_greedy',
                                                       'boltzmann'], default='semi_uniform')

# ========== MODEL PARAMETERS
parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
parser.add_argument('--advantage', choices=['naive', 'max', 'avg'], default='naive')
parser.add_argument('--memory_steps', type=int, default=0,
                    help='Do you want the state-action history to be part of the current state? How many steps?')
parser.add_argument('--rnn_steps', type=int, default=5)
parser.add_argument('--rnn', action='store_true', default=False)
parser.add_argument('--hidden_size', default='[20,20]',
                    help='Input the size of the network in a form of a list.\n'
                         'Example: --hidden_size [30,20] \n'
                         'This produces a network with two hidden layers of sizes 30 and 20. Don\'t use whitespace!')

# ========== OTHER PARAMETERS
parser.add_argument('environment', help='The gym environment, for autonomous vehicles, use ACar-v0')
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--display', action='store_true', default=True)
parser.add_argument('--no_display', dest='display', action='store_false')
parser.add_argument('--gym_record')
parser.add_argument('--wait', type=float, default=.015)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--save_path', type=str, default='models')
parser.add_argument('--mode', choices=['train', 'play', 'vtrain', 'test'], default='train',
                    help='''train: begin training, you should specify where to save the learned models(--save_path). \n
                            vtrain: training with visuals (for ACar, otherwise use --train --display).\n
                            test: test on a learned model, requires --load_path .\n
                            play: visual test (for ACar, otherwise use --test --display).''')
parser.add_argument('--load_path')
parser.add_argument('--result_id')

parser.add_argument('--save_frequency', type=int, default=400)
parser.add_argument('--test_episodes', type=int, default=40)
parser.add_argument('--save_models', action="store_true", default=True)
parser.add_argument('--dont_save_models', action="store_false", dest='save_models')


args = parser.parse_args()

assert isinstance(args.hidden_size, str)
args.hidden_size = eval(args.hidden_size)

if not os.path.exists(args.save_path):
    print 'Creating model directory', args.save_path
    os.makedirs(args.save_path)


def train(dddpg):
    total_reward = 0
    total_rewards = []
    step = 0
    learning_steps = 1

    best_reward = -999.
    exploration = get_strategy(args.exploration_strategy)

    for i_episode in range(args.episodes):
        observation = get_state(reset_environment())
        episode_reward = 0
        exploration.update(args)
        for t in range(args.max_timesteps):

            if test_now(i_episode):
                q = dddpg.target_model.predict(np.array([observation]), batch_size=1)
            else:
                q = dddpg.model.predict(np.array([observation]), batch_size=1)
            action = exploration.sample(q[0])
            if args.verbose > 0:
                print("e:", i_episode, "e.t:", t, "action:", action, "q:", q)

            replay_holder = ReplayHolder(observation, action)

            observation, reward, done, info = env.step(action)
            observation = get_state(observation)

            replay_holder.complete(reward, observation, done)
            dddpg.add_to_replay(replay_holder, step >= args.training_start_size)

            episode_reward += reward
            if args.verbose > 1:
                print("reward:", reward)

            step += 1
            if step == args.training_start_size and args.verbose >= 0:
                print bcolors.OKBLUE + 'TRAINING STARTED' + bcolors.ENDC

            if step > args.training_start_size and i_episode % args.save_frequency >= 10:
                if step % args.update_frequency == 0:
                    for k in xrange(args.train_repeat):
                        dddpg.train_on_batch()
                        learning_steps += 1
                        if args.prioritize and learning_steps % args.update_tree_interval == 0:
                            dddpg.heap_update()

                if step % args.target_net_update_frequency == 0:
                    weights = dddpg.model.get_weights()
                    dddpg.target_model.set_weights(weights)

            if done:
                break

        episode_print = "Episode {} finished after {} timesteps, episode reward {}".format(i_episode + 1, t + 1, episode_reward)
        if episode_reward > 0 and args.verbose >= 0:
            print bcolors.OKGREEN + episode_print + bcolors.ENDC
        elif args.verbose >= 0:
            print episode_print

        total_reward += episode_reward
        total_rewards.append(episode_reward)

        if i_episode % args.save_frequency == args.test_episodes-1:
            avg_r = float(np.mean(total_rewards[-(args.test_episodes-1):]))
            if args.verbose >= 0:
                print bcolors.YELLOW + 'Average reward (after %i learning steps): %.2f (best is %.2f)' % (learning_steps, avg_r, best_reward) + bcolors.ENDC
            folder_name = args.environment+'_'+str(i_episode)+str('_%.2f' % avg_r)
            if args.save_models:
                dddpg.save(args.save_path, folder_name)
            if avg_r > best_reward:
                best_reward = avg_r
                if args.save_models:
                    dddpg.save(args.save_path, 'best')

    print_results(best_reward, args)

    if args.gym_record:
        env.monitor.close()


def play(dddpg):
    if args.gym_record:
        env.monitor.start(directory=args.gym_record,
                          video_callable=lambda count: count % 1 == 0)
    total_reward = 0
    timestep = 0
    exploration = get_strategy(args.exploration_strategy, play=True)
    stuck = 0.
    successful = 0.
    for i_episode in range(args.episodes):
        observation = get_state(reset_environment())
        episode_reward = 0
        for t in range(args.max_timesteps):
            if args.display:
                env.render()

            if args.mode == 'play':
                time.sleep(args.wait)
            s = np.array([observation])
            q = dddpg.target_model.predict(s, batch_size=1)

            action = exploration.sample(q[0])
            if args.verbose > 0:
                print("e:", i_episode, "e.t:", t, "action:", action, "q:", q)

            observation, reward, done, info = env.step(action)
            observation = get_state(observation)
            episode_reward += reward
            if args.verbose > 1:
                print("reward:", reward)

            timestep += 1
            if done and info.get('success'):
                break

        episode_print = "Episode {} finished after {} timesteps, episode reward {}".format(i_episode + 1, t + 1,
                                                                                           episode_reward)

        if env.success:
            print bcolors.OKGREEN + episode_print + bcolors.ENDC
            successful += 1.
        else:
            print episode_print
            stuck += 1.
        total_reward += episode_reward

    print("Average reward per episode {}".format(total_reward / args.episodes))
    print("{}% success,  {}% stuck".format(100*successful / args.episodes, 100*stuck / args.episodes))



def test_now(i):
    """True if the agent should act according to the target network (no training)"""
    return i % args.save_frequency < 10


def get_state(obs):
    global full_state
    if args.rnn:
        full_state = shift(full_state, (1, 0))
        full_state[0] = obs
        return full_state
    else:
        return obs


def reset_environment():
    global full_state
    full_state = np.zeros_like(full_state)
    return env.reset()


if __name__ == '__main__':
    np.random.seed(args.seed)
    env = gym.make(args.environment)
    if 'ACar' in args.environment:
        env.configure(args)
    args.observation_space_shape = env.observation_space.shape
    args.action_space_size = env.action_space.n

    if args.load_path:
        dddpg = load(args.load_path)
    else:
        dddpg = DuelingModel(args)

    if args.rnn:
        full_state = np.zeros((args.rnn_steps,) + env.observation_space.shape)
    else:
        full_state = np.zeros(env.observation_space.shape)

    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)

    if args.gym_record:
        env.monitor.start(args.gym_record, force=True)

    if 'train' in args.mode:
        train(dddpg)
    else:
        play(dddpg)

