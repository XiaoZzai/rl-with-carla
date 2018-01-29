#!/usr/bin/python

# env parameters
MAX_EPISODES = 200000
MAX_EP_STEPS = 200000

# learning parameters

LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 80000

BATCH_SIZE = 32
TARGET_REPLACE_STEP = 1000

EXPLORATION = 2000000
EPS_INIT = 0.4
EPS_FINNAL = 0.001
OBSERVE = 33

