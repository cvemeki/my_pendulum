# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:06:04 2020

@author: tzhang
"""

import numpy as np
import dynamicProgramming
# from deepQLearning import dqnAgentCart
from deepQLearning import dqnAgent

from QLearning import qAgent
import QLearning

import matplotlib.pyplot as plt

import seaborn as sns


if __name__ == '__main__':
    
    '''********** basic and dummy agent **************'''
    
    '''dummy agent'''
    cartDummy = dqnAgent("CartPole-v0")
    cartDummy.build()
    cartDummy.offlineScore()
    
    
    '''basic agent'''
    cart0 = dqnAgent("CartPole-v0")
    cart0.build()
    cart0.learning("random")
    cart0.offlineScore()
    
    
    '''offline evaluation'''
    ax = plt.axes()
    ax = sns.distplot(cartDummy.scoreTable, axlabel="test score", color = 'gray', label = 'dummy agent')
#    ax = sns.distplot(cart0.scoreTable, axlabel="test score", color = 'green', label = 'basic agent')
    ax.set_ylabel("number of tests (density)")
    ax.axvline(x=0.975, linestyle = ':', color = 'r', label = "score considered as solved")
    plt.legend()
    plt.show()
    
    
    '''online evaluation'''
    ax = plt.axes()
    plt.plot(cart0.scoreEpisodes, color = "green", label = 'basic agent')
    ax.axhline(y=0.975, linestyle = ':', color = 'r', label = "score considered as solved")
    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score")
    plt.show()
    
    '''********** EXP1: explore rate **************'''
    
    cart1 = dqnAgent("CartPole-v0")   # this one can be overfitting!!
    cart1.EXPLORE_DECAY = 0.95
    cart1.build()
    cart1.learning("random")
    cart1.offlineScore()
    
    cart2 = dqnAgent("CartPole-v0")  # this converges very fast!!!
    cart2.EXPLORE_DECAY = 0.8
    cart2.build()
    cart2.learning("random")
    cart2.offlineScore()
    
    cart211 = dqnAgent("CartPole-v0")  # this converges very fast!!!
    cart211.EXPLORE_DECAY = 0.8
    cart211.CONSECUTIVE_EPISODE = 400
    cart211.build()
    cart211.learning("random")
    cart211.offlineScore()
    
    cart212 = dqnAgent("CartPole-v0")  # this converges very fast!!!
    cart212.EXPLORE_DECAY = 0.8
    cart211.CONSECUTIVE_EPISODE = 400
    cart212.EPISODES_PER_LEARNING_MAX = 50
    cart212.build()
    cart212.learning("random")
    cart212.offlineScore()
    

    '''offline evaluation'''
    ax = plt.axes()
#    ax = sns.distplot(cartDummy.scoreTable, axlabel="test score", color = 'gray', label = 'dummy agent')
#    ax = sns.distplot(cart0.scoreTable, axlabel="test score", color = 'green', label = 'basic agent', kde = 1, hist = 0)
#    ax = sns.distplot(cart1.scoreTable, axlabel="test score", color = 'blue', label = 'agent with explore decay = 0.95')
#    ax = sns.distplot(cart2.scoreTable, axlabel="test score", color = 'orange', label = 'agent with explore decay = 0.8', rug = 1)
    ax = sns.distplot(cart211.scoreTable, axlabel="test score", color = 'gray', label = 'agent with explore decay = 0.8, 500 episodes')
    ax = sns.distplot(cart212.scoreTable, axlabel="test score", color = 'orange', label = 'agent with explore decay = 0.8, 50 episodes')
    ax.set_ylabel("number of tests (density)")
#    plt.ylim(0,500)
    ax.axvline(x=0.975, linestyle = ':', color = 'r', label = "score considered as solved")
    plt.legend()
    plt.show()
    
    
    '''online evaluation'''
    ax = plt.axes()
#    plt.plot(cart0.scoreEpisodes, color = "green", label = 'basic agent with explore decay = 0.88')
#    plt.plot(cart1.scoreEpisodes[:110], color = "blue", label = 'agent with explore decay = 0.95')
#    plt.plot(cart2.scoreEpisodes, color = "orange", label = 'agent with explore decay = 0.8')
    plt.plot(cart211.scoreEpisodes, color = "gray", label = 'agent with explore decay = 0.8, 500 episodes')
    plt.plot(cart212.scoreEpisodes, color = "orange", label = 'agent with explore decay = 0.8, 50 episodes')
    ax.axhline(y=0.975, linestyle = ':', color = 'r', label = "score considered as solved")
    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score")
    plt.show()
    
    '''********** EXP2: batch size **************'''
    
    cart3 = dqnAgent("CartPole-v0")
    cart3.BATCH_SIZE = 200
    cart3.build()
    cart3.learning("random")
    cart3.offlineScore()
    
    cart4 = dqnAgent("CartPole-v0")
    cart4.BATCH_SIZE = 1
    cart4.EPISODES_PER_LEARNING_MAX = 2000
    cart4.build()
    cart4.learning("random")
    cart4.offlineScore()
    
    '''offline evaluation'''
    ax = plt.axes()
#    ax = sns.distplot(cartDummy.scoreTable, axlabel="test score", color = 'gray', label = 'dummy agent')
#    ax = sns.distplot(cart0.scoreTable, axlabel="test score", color = 'green', label = 'basic agent', kde = 1, hist = 1)
    ax = sns.distplot(cart3.scoreTable, axlabel="test score", color = 'blue', label = 'agent with batch size = 200')
    ax = sns.distplot(cart4.scoreTable, axlabel="test score", color = 'orange', label = 'agent with batch size = 1')
    ax.set_ylabel("number of tests (density)")
#    plt.ylim(0,500)
    ax.axvline(x=0.975, linestyle = ':', color = 'r', label = "score considered as solved")
    plt.legend()
    plt.show()
    
    
    '''online evaluation'''
    ax = plt.axes()
    plt.plot(cart0.scoreEpisodes, color = "green", label = 'basic agent, with batch size = 20')
    plt.plot(cart3.scoreEpisodes[:110], color = "blue", label = 'agent with batch size = 200')
    plt.plot(cart4.scoreEpisodes, color = "orange", label = 'agent with batch size = 1')
    ax.axhline(y=0.975, linestyle = ':', color = 'r', label = "score considered as solved")
    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score")
    plt.show()
    
