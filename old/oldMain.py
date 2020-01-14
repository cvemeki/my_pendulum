# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:15:25 2020

@author: tzhang
"""

from old import test
import numpy as np
import dynamicProgramming
# from deepQLearning import dqnAgentCart
from deepQLearning import dqnAgent

from QLearning import qAgent
import QLearning

# for cartpole
# agentCart = dqnAgentCart()
# agentCart.learning(([np.pi, 0], [np.pi, 0]))


# for pendulum
# agent = dqnAgent("Pendulum-v0")
# agent.BATCH_SIZE = 200
# agent.EPISODES_PER_LEARNING_MAX = 2
# agent.learning("random")

# agent = dqnAgent("CartPole-v0")
# agent.EPISODES_PER_LEARNING_MAX = 1
# agent.learning("random")

# for pendulum Q table learning
# agentPend = qAgentPend()
# agentPend.learning("random")

# test 1, fail
# agent = dqnAgent("Pendulum-v0")
# agent.BATCH_SIZE = 1
# agent.GAMMA = 0.8
# agent.EPISODES_PER_LEARNING_MAX = 2000
# agent.learning("random")

# test 2, -- fail
# agent = dqnAgent("Pendulum-v0")
# agent.BATCH_SIZE = 20
# agent.GAMMA = 0.95
# agent.LEARNING_RATE = 0.8
# agent.EPISODES_PER_LEARNING_MAX = 2000
# agent.learning("random")

# test 3, -- fail
# agent = dqnAgent("Pendulum-v0")
# agent.BATCH_SIZE = 20
# agent.STEPS_PER_EPISODE_MAX = 50
# agent.GAMMA = 0.95
# agent.LEARNING_RATE = 0.8
# agent.EPISODES_PER_LEARNING_MAX = 2000
# agent.learning("random")

# test 4, --  success! with random choose equal action
#agent = dqnAgent("Pendulum-v0")
#agent.BATCH_SIZE = 200
#agent.STEPS_PER_EPISODE_MAX = 200
#agent.GAMMA = 0.95
#agent.LEARNING_RATE = 0.8
#agent.EPISODES_PER_LEARNING_MAX = 100
#agent.learning("random")

# test 5, -- no result
# agent = dqnAgent("Pendulum-v0")
# agent.BATCH_SIZE = 200
# agent.STEPS_PER_EPISODE_MAX = 200
# agent.GAMMA = 0.95
# agent.LEARNING_RATE = 1
# agent.EPISODES_PER_LEARNING_MAX = 100
# agent.learning("random")

# test 6 -- succeed! without random choose equal action
# agent = dqnAgent("Pendulum-v0")
# agent.BATCH_SIZE = 200
# agent.STEPS_PER_EPISODE_MAX = 200
# agent.GAMMA = 0.8
# agent.EXPLORE_RATE_INIT = 0.5
# agent.EXPLORE_RATE = 0.5
# agent.EXPLORE_DECAY = 0.999
# agent.LEARNING_RATE = 0.99
# agent.EPISODES_PER_LEARNING_MAX = 100
# agent.learning("random")

# test 7 -- interupted. 0.01 at top
# agent = dqnAgent("Pendulum-v0")
# agent.BATCH_SIZE = 200
# agent.STEPS_PER_EPISODE_MAX = 200
# agent.GAMMA = 0.8
# agent.EXPLORE_RATE_INIT = 0.5
# agent.EXPLORE_RATE = 0.5
# agent.EXPLORE_DECAY = 0.999
# agent.LEARNING_RATE = 0.99
# agent.EPISODES_PER_LEARNING_MAX = 100
# agent.learning("random")


# test 8 -- interupted. 0.01 at top, with immediat replay
# agent = dqnAgent("Pendulum-v0")
# agent.BATCH_SIZE = 200
# agent.STEPS_PER_EPISODE_MAX = 200
# agent.GAMMA = 0.8
# agent.EXPLORE_RATE_INIT = 0.5
# agent.EXPLORE_RATE = 0.5
# agent.EXPLORE_DECAY = 0.999
# agent.LEARNING_RATE = 0.99
# agent.EPISODES_PER_LEARNING_MAX = 100
# agent.learning("random")

# # test 9 -- not learning 0.1 at top, without immediat replay
# agent = qAgent("Pendulum-v0")
# agent.learning("random")

# test 10 -- 0.1 at top, without immediat replay
#    agent = qAgent("Pendulum-v0")
#    agent.EXPLORE_DECAY = 0.99999
#    agent.learning("random")

# test 11 interupt -- binary action  decrease state space
# agent = qAgent("Pendulum-v0")
# agent.EXPLORE_DECAY = 0.99995
# agent.learning("random")

# learning rate = 0.8

if __name__ == '__main__':
    testP1 = 1
#    '''smaller final bonus range'''
#    agent2 = dqnAgent("Pendulum-v0")
#    agent2.terminalBonusRange = 0.05
#    agent2.build()
#    agent2.learning("random")