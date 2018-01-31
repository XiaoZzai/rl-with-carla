#!/usr/bin/python

import tensorflow as tf
import numpy as np
import math
from config import *

###############################  ddpg  ####################################

class ddpg(object):
    def __init__(self, s_dim, a_dim, trainflag=True):

        self.trainflag = trainflag
        self.epsilon = EPS_INIT

        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.target_replace_counter = 0

        self.log_loss_step   = 0
        self.log_reward_step = 0

        self.a_dim = a_dim
        self.s_dim = s_dim

        self.S  = tf.placeholder(tf.float32, [None, s_dim], name='state')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], name='next_state')
        self.R  = tf.placeholder(tf.float32, [None, 1],     name='reward')

        with tf.variable_scope('actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)

        with tf.variable_scope('critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/target')

        # target net replacement
        with tf.name_scope("replace_target"):
            self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                                 for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        with tf.name_scope("q_target"):
            q_target = self.R + GAMMA * q_

        with tf.name_scope("critic_train"):
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        with tf.name_scope("actor_train"):
            a_loss = - tf.reduce_mean(q)
            self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        # Saver
        self.saver = tf.train.Saver(max_to_keep=10)

        # Log
        self.r = tf.placeholder(tf.float32)
        with tf.name_scope("tensorboard"):
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
            self.loss_critic = tf.summary.scalar("critic_loss", td_error)
            self.loss_actor = tf.summary.scalar("actor_loss", a_loss)
            self.reward = tf.summary.scalar("reward", self.r)
            # self.merged  = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):

        action = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        if self.trainflag == True:
            if np.random.randn(1) < self.epsilon:
                last_rel_x = s[-2]
                last_rel_y = s[-1]
                rel_x = s[3]
                rel_y = s[4]
                self_speed = s[0]
                npc_speed = s[1]
                steer = 0.5 - math.atan2(last_rel_y * 40, last_rel_x * 4) / 3.141592653
                if abs(steer) < 0.15:
                    steer /= 2

                throttle = npc_speed - self_speed
                distance = math.sqrt((rel_x * 4)**2 + (rel_y * 40)**2)
                if distance > 40:
                    if throttle > 0:
                        throttle *= 12
                    else:
                        throttle = - throttle * 6
                elif distance > 20:
                    if throttle > 0:
                        throttle *= 8
                    else:
                        throttle = - throttle * 4
                else:
                    if throttle < 0:
                        throttle *= 2

                action = [steer, throttle]
            else:
                # add noise
                noise = np.zeros(self.a_dim)
                noise[0] = self.epsilon * 0.05 * np.random.randn(1)
                if action[1] < 0:
                    noise[1] = -self.epsilon * 0.2 * np.random.randn(1)
                else:
                    noise[1] = self.epsilon * 0.1 * np.random.randn(1)
                action += noise

        action = [np.clip(action[0], -1.0, 1.0), np.clip(action[1], -1.0, 1.0)]

        if (self.pointer >= OBSERVE) and (self.pointer < EXPLORATION + OBSERVE):
            self.epsilon -= (EPS_INIT - EPS_FINNAL) / EXPLORATION

        return action

    def learn(self):

        if self.pointer >= OBSERVE and self.trainflag==True:
            if self.pointer < MEMORY_CAPACITY:
                trans_counter = self.pointer
            else:
                trans_counter = MEMORY_CAPACITY

            indices = np.random.choice(trans_counter, size=BATCH_SIZE)
            bt  = self.memory[indices, :]
            bs  = bt[:, :self.s_dim]
            ba  = bt[:, self.s_dim: self.s_dim + self.a_dim]
            br  = bt[:, -self.s_dim - 1: -self.s_dim]
            bs_ = bt[:, -self.s_dim:]

            self.sess.run(self.atrain, {self.S: bs})
            self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

            if self.pointer % 10 == 0:
                results = self.sess.run(self.loss_critic, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
                self.writer.add_summary(results, self.log_loss_step)

                # results = self.sess.run(self.loss_actor, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
                # self.writer.add_summary(results, self.log_loss_step)

                self.log_loss_step += 1

            # soft target replacement every "TARGET_REPLACE_STEP"
            self.target_replace_counter += 1
            if self.target_replace_counter % TARGET_REPLACE_STEP == 0:
                self.sess.run(self.soft_replace)

    def store_transition(self, s, a, r, s_, eps_reward=0):
        if eps_reward != 0:
            results = self.sess.run(self.reward, {self.r : eps_reward})
            self.writer.add_summary(results, self.log_reward_step)
            self.log_reward_step += 1

        if self.trainflag == True:
            transition = np.hstack((s, a, [r], s_))
            index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
            self.memory[index, :] = transition
            self.pointer += 1

    def save(self, step):
        # saving networks
        self.saver.save(self.sess, 'model/', global_step=step)

        # saving transition
        np.save("model/ddpg-memory.npy", self.memory)
        with open("model/pointer", 'w') as file:
            file.write(str(self.pointer))

    def load(self):
        # loading networks
        checkpoint = tf.train.get_checkpoint_state("model")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        # load transition
        self.memory = np.load("model/ddpg-memory.npy")
        with open("model/pointer", 'r') as file:
            self.pointer = int(file.readline())

        if self.pointer >= OBSERVE and self.pointer < EXPLORATION + OBSERVE:
            self.epsilon -= (self.pointer - OBSERVE) * ((EPS_INIT - EPS_FINNAL) / EXPLORATION)

    def _build_a(self, state, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 256
            n_l2 = 512
            l1 = tf.layers.dense(state, n_l1, activation=tf.nn.relu, name='layer1', trainable=trainable)
            l2 = tf.layers.dense(l1, n_l2, activation=tf.nn.relu, name='layer2', trainable=trainable)
            action  = tf.layers.dense(l2, self.a_dim, activation=tf.nn.tanh, name='action', trainable=trainable)
            return action

    def _build_c(self, state, action, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 256
            n_l2 = 512
            w1_s = tf.get_variable('layer1_weight_state', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('layer1_weight_action', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            l1 = tf.nn.relu(tf.matmul(state, w1_s) + tf.matmul(action, w1_a) + b1, name='layer1')
            l2 = tf.layers.dense(l1, n_l2, activation=tf.nn.relu, name='layer2', trainable=trainable)
            critic = tf.layers.dense(l2, 1, name='q_value', trainable=trainable)
            return critic

