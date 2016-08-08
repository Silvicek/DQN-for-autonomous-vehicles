import argparse
import gym
from gym.spaces import Box, Discrete
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras import initializations
from keras import backend as K
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--batch_norm', action="store_true", default=False)
parser.add_argument('--no_batch_norm', action="store_false", dest='batch_norm')
parser.add_argument('--replay_start_size', type=int, default=5000)
parser.add_argument('--train_repeat', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.99)
# parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--episodes', type=int, default=20000)
parser.add_argument('--max_timesteps', type=int, default=2000)
parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
# parser.add_argument('--optimizer_lr', type=float, default=0.001)
parser.add_argument('--exploration', type=float, default=0.1)  # TODO: remove?
parser.add_argument('--advantage', choices=['naive', 'max', 'avg'], default='naive')
parser.add_argument('--display', action='store_true', default=True)
parser.add_argument('--no_display', dest='display', action='store_false')
parser.add_argument('--gym_record')
parser.add_argument('--update_frequency', type=int, default=4)
parser.add_argument('--target_net_update_frequency', type=int, default=32)
parser.add_argument('--replay_memory_size', type=int, default=100000)

parser.add_argument('--save_frequency', type=int, default=100)
parser.add_argument('--save_path', type=str, default='models')
parser.add_argument('--mode', choices=['train', 'play', 'vtrain'], default='train')
parser.add_argument('--load_path')

parser.add_argument('environment')

args = parser.parse_args()


def create_models():
    x, z = createLayers()
    model = Model(input=x, output=z)
    model.summary()
    model.compile(optimizer=args.optimizer, loss='mse')

    x, z = createLayers()
    target_model = Model(input=x, output=z)

    if args.load_path is not None:
        model.load_weights(args.load_path)
    target_model.set_weights(model.get_weights())

    return model, target_model


def createLayers():
    custom_init = lambda shape, name: initializations.normal(shape, scale=0.01, name=name)
    x = Input(shape=env.observation_space.shape)
    if args.batch_norm:
        h = BatchNormalization()(x)
    else:
        h = x
    for i in range(args.layers):
        h = Dense(args.hidden_size, activation=args.activation, init=custom_init)(h)
        if args.batch_norm and i != args.layers - 1:
            h = BatchNormalization()(h)
    y = Dense(env.action_space.n + 1)(h)
    if args.advantage == 'avg':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                   output_shape=(env.action_space.n,))(y)
    elif args.advantage == 'max':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                   output_shape=(env.action_space.n,))(y)
    elif args.advantage == 'naive':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:], output_shape=(env.action_space.n,))(y)
    else:
        assert False

    return x, z





def update_exploration(e):  # TODO: dynamic change
    if e <= args.exploration:
        return e
    else:
        return e - 2./args.episodes


def train():
    prestates = []
    actions = []
    rewards = []
    poststates = []
    terminals = []

    total_reward = 0
    timestep = 0
    learning_steps = 0
    epsilon = 1.

    best_reward = -999.  # TODO: save THE BEST THE BEST THE BEST THE BEST

    for i_episode in range(args.episodes):
        observation = env.reset()
        episode_reward = 0
        epsilon = update_exploration(epsilon)
        for t in range(args.max_timesteps):
            if args.display:
                env.render()

            if i_episode % args.save_frequency >= 10 and \
               (timestep < args.replay_start_size or np.random.random() < epsilon):
                action = env.action_space.sample()
                if args.verbose > 0:
                    print("e:", i_episode, "e.t:", t, "action:", action, "random")
            else:
                s = np.array([observation])
                if i_episode % args.save_frequency < 10:
                    q = target_model.predict(s, batch_size=1)
                else:
                    q = model.predict(s, batch_size=1)
                action = np.argmax(q[0])
                if args.verbose > 0:
                    print("e:", i_episode, "e.t:", t, "action:", action, "q:", q)

            if len(prestates) >= args.replay_memory_size:
                delidx = np.random.randint(0, len(prestates) - 1 - args.batch_size)
                del prestates[delidx]
                del actions[delidx]
                del rewards[delidx]
                del poststates[delidx]
                del terminals[delidx]

            prestates.append(observation)
            actions.append(action)

            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if args.verbose > 1:
                print("reward:", reward)

            rewards.append(reward)
            poststates.append(observation)
            terminals.append(done)

            timestep += 1

            if timestep > args.replay_start_size and i_episode % args.save_frequency >= 10:
                if timestep % args.update_frequency == 0:
                    for k in xrange(args.train_repeat):
                        if len(prestates) > args.batch_size:
                            # indexes = range(args.batch_size)
                            # indexes = np.random.choice(len(prestates), size=args.batch_size)
                            indexes = np.random.randint(len(prestates), size=args.batch_size)
                        else:
                            indexes = range(len(prestates))

                        pre_sample = np.array([prestates[i] for i in indexes])
                        post_sample = np.array([poststates[i] for i in indexes])
                        qpre = model.predict(pre_sample)
                        qpost = target_model.predict(post_sample)
                        for i in xrange(len(indexes)):
                            if terminals[indexes[i]]:
                                qpre[i, actions[indexes[i]]] = rewards[indexes[i]]
                            else:
                                qpre[i, actions[indexes[i]]] = rewards[indexes[i]] + args.gamma * np.amax(qpost[i])
                        model.train_on_batch(pre_sample, qpre)
                        learning_steps += 1

                if timestep % args.target_net_update_frequency == 0:
                    if args.verbose > 0:
                        print('learned on batch:', learning_steps, 'DDQN: Updating weights')
                    weights = model.get_weights()
                    target_model.set_weights(weights)
                        # weights = model.get_weights()
                        # target_weights = target_model.get_weights()
                        # for i in xrange(len(weights)):
                        #     weights[i] *= args.tau
                        #     target_weights[i] *= (1 - args.tau)
                        #     target_weights[i] += weights[i]
                        # target_model.set_weights(target_weights)

            if done:
                break

        if i_episode % args.save_frequency == 0:
            file_name = args.environment+'_'+str(i_episode)+str('_%.2f' % episode_reward)
            target_model.save_weights(args.save_path+'/'+file_name)

        print("Episode {} finished after {} timesteps, episode reward {}".format(i_episode + 1, t + 1, episode_reward))
        total_reward += episode_reward

    print("Average reward per episode {}".format(total_reward / args.episodes))

    if args.gym_record:
        env.monitor.close()


def play():
    if args.gym_record:
        env.monitor.start(directory=args.gym_record,
                          video_callable=lambda count: count % 1 == 0)
    total_reward = 0
    timestep = 0
    for i_episode in range(args.episodes):
        observation = env.reset()
        episode_reward = 0
        for t in range(args.max_timesteps):
            if args.display:
                env.render()
            # import time
            # time.sleep(1)
            s = np.array([observation])
            q = target_model.predict(s, batch_size=1)
            action = np.argmax(q[0])
            if args.verbose > 0:
                print("e:", i_episode, "e.t:", t, "action:", action, "q:", q)

            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if args.verbose > 1:
                print("reward:", reward)

            timestep += 1

            if done:
                break

        print("Episode {} finished after {} timesteps, episode reward {}".format(i_episode + 1, t + 1, episode_reward))
        total_reward += episode_reward

    print("Average reward per episode {}".format(total_reward / args.episodes))


if __name__ == '__main__':
    env = gym.make(args.environment)
    env.configure(args.mode)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)

    if args.gym_record:
        env.monitor.start(args.gym_record, force=True)

    model, target_model = create_models()
    if 'train' in args.mode:
        train()
    else:
        play()

