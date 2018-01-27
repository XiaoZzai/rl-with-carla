#!/usr/bin/python

import numpy as np
import traceback

from gym_carla import gym_carla_car_following
from ddpg import ddpg

# observe_step_max = 10000

def interactive_with_environment(agent, env):

    # Environment parameters

    MAX_EPISODES = 200000
    MAX_EP_STEPS = 200000

    # Statistics
    total_episode = 0
    total_step    = 0
    each_episode_step   = np.zeros(MAX_EPISODES)
    each_episode_reward = np.zeros(MAX_EPISODES)

    for i in range(MAX_EPISODES):

        print("Episode %d starts " % i)

        state = env.reset()

        for j in range(MAX_EP_STEPS):

            # main algorithm
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            env.print_state(total_step, state, action, reward)

            agent.store_transition(state, action, reward, next_state)
            agent.learn()

            state                   = next_state
            total_step             += 1
            each_episode_step[i]   += 1
            each_episode_reward[i] += reward

            if done == True:
                env.stop()
                break

        total_episode += 1
        print("Episode %d ends with %d steps (reward = %f) " % (i, each_episode_step[i], each_episode_reward[i]))

        if i % 200 == 199:
            agent.save(i)


# def store_experience(agent, env):
#
#     observe_step = 0
#
#     observe_step_turn     = int(observe_step_max * 0.5)
#     observe_step_straight = int(observe_step_max * 0.5)
#
#     total_step   = 0
#     print("Begin Store Experiences !")
#
#     states  = np.array([observe_step_max, env.observation_space[0]])
#     actions = np.array([observe_step_max, env.action_space[0]])
#
#     action = [0.0, 0.0]
#     while True:
#
#         if observe_step >= observe_step_max:
#             break
#
#         state = env.reset()
#         j = 0
#         while True:
#             #
#             delta_speed = state[0] - state[1]
#             if delta_speed < 0:
#                 if state[3] >= 0.5:
#                     action[1] = 0.65
#                 else:
#                     action[1] = 0.35
#             else:
#                 if state[3] >= 0.5:
#                     action[1] = 0
#                 else:
#                     action[1] = -0.5
#
#             next_state, reward, done, info = env.step(action)
#             # import matplotlib.pyplot as plt
#             # plt.imshow(state.reshape([128, 128]), cmap="gray")
#             # plt.show()
#             if (abs(action[0]) > 0.05) and (observe_step_turn > 0):
#                 agent.store_experience(state, action, reward, next_state, done)
#                 states[observe_step] = state
#                 observe_step_turn -= 1
#                 observe_step += 1
#             else:
#                 if ((j > 50) and (observe_step < observe_step_max) and (total_step % 10 == 0) and (observe_step_straight > 0)):
#                     agent.store_experience(state, action, reward, next_state, done)
#                     states[observe_step] = state
#                     observe_step_straight -= 1
#                     observe_step += 1
#
#             if total_step % 100 == 0:
#                 print(observe_step, observe_step_turn, observe_step_straight)
#
#             total_step   += 1
#             j += 1
#
#             if (done == True) or (observe_step >= observe_step_max):
#                 break
#
#     agent.shuffle_buffer()
#     np.save("states.npy", states)
#     np.save("actions.npy", actions)
#     print("End Store Experiences With %d !" % (observe_step))
#
# def train(agent, env):
#     print("Begin Training Model")
#
#     states = np.load("states.npy")
#     actions = np.load("actions.npy")
#
#     from model import create_actor_model
#     model, _, _ = create_actor_model(env.observation_space, env.action_space)
#
#     xs_train = states[:observe_step_max * 0.8]
#     ys_train = actions[:observe_step_max * 0.8]
#     xs_test = states[-observe_step_max * 0.2:]
#     ys_test = actions[-observe_step_max * 0.2:]
#     model.fit(xs_train, ys_train, 32, 1000, verbose=2)
#     print("Accuracy %f " % (model.evaluate(xs_test, ys_test, verbose=2)))
#
#     # for i in range(20000):
#     #     agent.train()
#     #     if i % 1000 == 999:
#     #         print("Train Iterator %d" % (i) )
#     #         agent.save_model()
#     print("End Training Model")

def main(trainable=True):

    import time

    env   = None
    agent = None
    while True:
        try:
            env   = gym_carla_car_following("127.0.0.1", 2000, 15)
            agent = ddpg(env.observation_space.shape[0], env.action_space.shape[0], trainable)

            try:
                agent.load()
                print("Load Params Success")
            except:
                traceback.print_exc()

            # if trainable == True:
            #     store_experience(agent, env)
            #     train(agent, env)

            interactive_with_environment(agent, env)
        except:
            traceback.print_exc()
        finally:
            # if env is not None:
            #     env.stop()
            if agent is not None:
                agent.save(-1)

            env = None
            agent = None
            import time
            time.sleep(5)


if __name__ == "__main__":
    main(True)
