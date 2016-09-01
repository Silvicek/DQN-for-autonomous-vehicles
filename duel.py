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
parser.add_argument('--train_repeat', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--episodes', type=int, default=20000)
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
parser.add_argument('--update_frequency', type=int, default=4)
parser.add_argument('--target_net_update_frequency', type=int, default=32)
parser.add_argument('--replay_size', type=int, default=100000)
parser.add_argument('--save_frequency', type=int, default=100)
parser.add_argument('--max_timesteps', type=int, default=1500)

parser.add_argument('--prioritize', action="store_true", default=False)
parser.add_argument('--update_tree_interval', type=int, default=100000)

parser.add_argument('--exploration_params', type=float, default=.1)
parser.add_argument('--exploration_strategy', choices=['semi_uniform', 'e_greedy',
                                                       'boltzmann'], default='semi_uniform')

# ========== MODEL PARAMETERS
parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
parser.add_argument('--advantage', choices=['naive', 'max', 'avg'], default='naive')
parser.add_argument('--memory_steps', type=int, default=3)
parser.add_argument('--rnn_steps', type=int, default=10)
parser.add_argument('--rnn', action='store_true', default=False)
parser.add_argument('--hidden_size', default='[20,20]')

# ========== OTHER PARAMETERS
# parser.add_argument('--environment', type=str, default='ACar-v0')
# parser.add_argument('--environment', type=str, default='CartPole-v0')
parser.add_argument('environment')
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--display', action='store_true', default=True)
parser.add_argument('--no_display', dest='display', action='store_false')
parser.add_argument('--gym_record')
parser.add_argument('--wait', type=float, default=.015)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--save_path', type=str, default='models')
parser.add_argument('--mode', choices=['train', 'play', 'vtrain', 'play2'], default='train')
parser.add_argument('--load_path')

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
            if step == args.training_start_size:
                print bcolors.OKBLUE + 'TRAINING STARTED' + bcolors.ENDC

            if step > args.training_start_size and i_episode % args.save_frequency >= 10:
                if step % args.update_frequency == 0:
                    for k in xrange(args.train_repeat):
                        dddpg.train_on_batch()
                        learning_steps += 1
                        if args.prioritize and learning_steps % args.update_tree_interval == 0:
                            dddpg.heap_update()

                if step % args.target_net_update_frequency == 0:
                    if args.verbose > 0:
                        print('learned on batch:', learning_steps, 'DDQN: Updating weights')
                    weights = dddpg.model.get_weights()
                    dddpg.target_model.set_weights(weights)

            if done:
                break

        episode_print = "Episode {} finished after {} timesteps, episode reward {}".format(i_episode + 1, t + 1, episode_reward)
        if episode_reward > 0:
            print bcolors.OKGREEN + episode_print + bcolors.ENDC
        else:
            print episode_print

        total_reward += episode_reward
        total_rewards.append(episode_reward)

        if i_episode % args.save_frequency == 9:
            avg_r = float(np.mean(total_rewards[-9:]))
            print bcolors.YELLOW + 'Average reward (after %i learning steps): %.2f (best is %.2f)' % (learning_steps, avg_r, best_reward) + bcolors.ENDC
            folder_name = args.environment+'_'+str(i_episode)+str('_%.2f' % avg_r)
            dddpg.save(args.save_path, folder_name)
            if avg_r > best_reward:
                best_reward = avg_r
                dddpg.save(args.save_path, 'best')

    print("Average reward per episode {}".format(total_reward / args.episodes))
    print_results(best_reward, args)

    if args.gym_record:
        env.monitor.close()


def play(dddpg):
    if args.gym_record:
        env.monitor.start(directory=args.gym_record,
                          video_callable=lambda count: count % 1 == 0)
    total_reward = 0
    timestep = 0
    exploration = get_strategy(args.exploration_strategy)
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

            if done:
                break

        episode_print = "Episode {} finished after {} timesteps, episode reward {}".format(i_episode + 1, t + 1,
                                                                                           episode_reward)
        if episode_reward > 0:
            print bcolors.OKGREEN + episode_print + bcolors.ENDC
        else:
            print episode_print
        total_reward += episode_reward

    print("Average reward per episode {}".format(total_reward / args.episodes))


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

