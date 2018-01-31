#!/usr/bin/python

import numpy as np
import traceback
from gym_carla import gym_carla_car_following
from ddpg import ddpg
from config import *

def interactive_with_environment(agent, env):

    # statistic
    total_episode = 0
    total_step = 0
    each_episode_step = np.zeros(MAX_EPISODES)
    each_episode_reward = np.zeros(MAX_EPISODES)

    for i in range(MAX_EPISODES):

        print("Episode %d starts " % i)

        state = env.reset()

        for j in range(MAX_EP_STEPS):

            # main algorithm
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            # env.print_state(total_step, state, action, reward, agent.epsilon)

            total_step += 1
            each_episode_step[i] += 1
            each_episode_reward[i] += reward

            if done == True:
                agent.store_transition(state, action, reward, next_state, each_episode_reward[i])
            else:
                agent.store_transition(state, action, reward, next_state)

            agent.learn()
            state = next_state

            if done == True:
                env.stop()
                break

        total_episode += 1
        print("Episode %d ends with %d steps (reward = %f) " % (i, each_episode_step[i], each_episode_reward[i]))

        if i % 500 == 499:
            agent.save(i)

def main(trainable=True):

    env = gym_carla_car_following("127.0.0.1", 2000, 15)
    agent = ddpg(env.observation_space.shape[0], env.action_space.shape[0], trainable)

    try:
        agent.load()
    except:
        traceback.print_exc()

    while True:
        try:
            interactive_with_environment(agent, env)
        except:
            traceback.print_exc()
        finally:
            agent.save(-1)

if __name__ == "__main__":
    main(True)
