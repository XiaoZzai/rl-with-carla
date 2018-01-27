#!/usr/bin/python

import tensorflow as tf
import numpy as np


#####################  hyper parameters  ####################

LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 40000

BATCH_SIZE = 32
TARGET_REPLACE_STEP = 1000

EXPLORATION = 2000000
EPS_INIT = 0.5
EPS_FINNAL = 0.001
OBSERVE = 30000

###############################  ddpg  ####################################

class ddpg(object):
    def __init__(self, s_dim, a_dim, trainflag=True):

        self.trainflag = trainflag
        self.epsilon = EPS_INIT

        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.target_replace_counter = 0

        self.a_dim = a_dim
        self.s_dim = s_dim

        self.S = tf.placeholder(tf.float32, [None, s_dim], name='s')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], name='s_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)

        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        # Saver
        self.saver = tf.train.Saver(max_to_keep=10)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):

        # add noise
        action = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        if self.trainflag == True:
            if np.random.randn(1) < self.epsilon:
                steer = self.epsilon / 2* np.random.randn(1) + 0.0
                throttle = self.epsilon / 2 * np.random.randn(1) + 0.6
                action[0] = steer
                action[1] = throttle
            else:
                noise = np.zeros(self.a_dim)
                noise[0] = self.epsilon * (0.6 * (0.0 - action[0]) + 0.3 * np.random.randn(1))
                if action[1] < 0:
                    noise[1] = -self.epsilon * (1.0 * (0.5 - action[1]) + 0.1 * np.random.randn(1))
                else:
                    noise[1] = self.epsilon * (1.0 * (-0.1 - action[1]) + 0.05 * np.random.randn(1))
                action += noise

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
            bt = self.memory[indices, :]
            bs = bt[:, :self.s_dim]
            ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
            br = bt[:, -self.s_dim - 1: -self.s_dim]
            bs_ = bt[:, -self.s_dim:]

            self.sess.run(self.atrain, {self.S: bs})
            self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

            # soft target replacement every "TARGET_REPLACE_STEP"
            self.target_replace_counter += 1
            if self.target_replace_counter % TARGET_REPLACE_STEP == 0:
                self.sess.run(self.soft_replace)

    def store_transition(self, s, a, r, s_):
        if self.trainflag == True:
            transition = np.hstack((s, a, [r], s_))
            index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
            self.memory[index, :] = transition
            self.pointer += 1

    def save(self, step):
        # saving networks
        self.saver.save(self.sess, 'param-ddpg/', global_step=step)

        # saving transition
        np.save("param-ddpg/ddpg-memory.npy", self.memory)
        with open("param-ddpg/pointer", 'w') as file:
            file.write(str(self.pointer))

    def load(self):
        # loading networks
        checkpoint = tf.train.get_checkpoint_state("param-ddpg")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        # load transition
        self.memory = np.load("param-ddpg/ddpg-memory.npy")
        with open("param-ddpg/pointer", 'r') as file:
            self.pointer = int(file.readline())

        if self.pointer >= OBSERVE and self.pointer < EXPLORATION + OBSERVE:
            self.epsilon -= (self.pointer - OBSERVE) * ((EPS_INIT - EPS_FINNAL) / EXPLORATION)

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 256
            n_l2 = 512
            l1 = tf.layers.dense(s, n_l1, activation=tf.nn.relu, name='l1', trainable=trainable)
            l2 = tf.layers.dense(l1, n_l2, activation=tf.nn.relu, name='l2', trainable=trainable)
            a  = tf.layers.dense(l2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 256
            n_l2 = 512
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            l1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1, name='l1')
            l2 = tf.layers.dense(l1, n_l2, activation=tf.nn.relu, name='l2', trainable=trainable)
            c = tf.layers.dense(l2, 1, name='c', trainable=trainable)
            return c

