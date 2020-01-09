from old import test
import numpy as np
import dynamicProgramming
# from deepQLearning import dqnAgentCart
from deepQLearning import dqnAgent

from QLearning import qAgentPend
import QLearning

# in the main we are going to run different methodes and do the comparisons.

if __name__ == '__main__':
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

    # test 4, --
    # agent = dqnAgent("Pendulum-v0")
    # agent.BATCH_SIZE = 200
    # agent.STEPS_PER_EPISODE_MAX = 200
    # agent.GAMMA = 0.95
    # agent.LEARNING_RATE = 0.8
    # agent.EPISODES_PER_LEARNING_MAX = 100
    # agent.learning("random")

    # test 5, --
    # agent = dqnAgent("Pendulum-v0")
    # agent.BATCH_SIZE = 200
    # agent.STEPS_PER_EPISODE_MAX = 200
    # agent.GAMMA = 0.95
    # agent.LEARNING_RATE = 1
    # agent.EPISODES_PER_LEARNING_MAX = 100
    # agent.learning("random")

    # test 6 --
    agent = dqnAgent("Pendulum-v0")
    agent.BATCH_SIZE = 200
    agent.STEPS_PER_EPISODE_MAX = 200
    agent.GAMMA = 0.8
    agent.EXPLORE_RATE_INIT = 0.5
    agent.EXPLORE_RATE = 0.5
    agent.EXPLORE_DECAY = 0.999
    agent.LEARNING_RATE = 0.99
    agent.EPISODES_PER_LEARNING_MAX = 100
    agent.learning("random")
