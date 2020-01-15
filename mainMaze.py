# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:18:37 2020

@author: tzhang
"""

from old import test
import numpy as np
import dynamicProgramming
# from deepQLearning import dqnAgentCart
from deepQLearning import dqnAgent

from QLearning import qAgent
import QLearning

import matplotlib.pyplot as plt

import seaborn as sns

import gym


if __name__ == '__main__':
    
    '''baseline maze model'''
    maze0 = qAgent("maze-sample-100x100-v0")
    maze0.EXPLORE_DECAY = 0.9998
    maze0.EXPLORE_MIN = 0.0
    maze0.build()
    maze0.learning("random")
    maze0.offlineScore()
    
    maze3 = qAgent("maze-sample-3x3-v0")
    maze3.EXPLORE_DECAY = 0.99
    maze3.build()
    maze3.learning("random")
    maze3.offlineScore()
    
    maze5 = qAgent("maze-sample-5x5-v0")
    maze5.EXPLORE_DECAY = 0.998
    maze5.build()
    maze5.learning("random")
    maze5.offlineScore()
    
    agent = maze10
    #    plt.plot(dummy.scoreEpisodes, label = "dummy agent", color = "grey")
    plt.plot(agent.scoreEpisodes, label = "score", color = "orange")
    plt.plot(agent.rewardEpisodes, label = "reward", color = "blue")
#    plt.plot(agent3.scoreEpisodes[:30], label = "with velocity weight = 0.1", color = "blue")
#    plt.plot(agent6.scoreEpisodes[:30], label = "with velocity weight = 10", color = "grey")

    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score/reward")
    
    
    maze10 = qAgent("maze-sample-10x10-v0")
    maze10.render = 0
    maze10.EXPLORE_DECAY = 0.998
    maze10.build()
    maze10.learning("random")
    maze10.offlineScore()
    
    
    maze0.plotQSurface(0,1,1)
    maze0.plotQSurface(0,2,5)
    maze0.plotQSurface(1,2,5)
    
    max_Q = np.amax(maze5.Q, axis=2)
    colormap = plt.cm.hot
    ax = sns.heatmap(max_Q, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()
    
    bestAction = np.argmax(maze0.Q, axis=2)
    bestAction = maze0.actionSpace[bestAction]
    colormap = plt.cm.hot
    ax = sns.heatmap(bestAction, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()