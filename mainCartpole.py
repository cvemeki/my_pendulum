# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:06:04 2020

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


if __name__ == '__main__':
    
    '''dummy agent'''
    cartDummy = dqnAgent("CartPole-v0")
    cartDummy.build()
    cartDummy.offlineScore()
    
    
    '''basic agent'''
    cart1 = dqnAgent("CartPole-v0")
    cart1.build()
    cart1.learning("random")
    cart1.offlineScore()
    
    
    '''offline evaluation'''
    ax = sns.distplot(cartDummy.scoreTable, axlabel="test score")
    ax = sns.distplot(cart1.scoreTable, axlabel="test score")
    ax.set_ylabel("number of tests")
    plt.show()
    
    
    '''online evaluation'''
    plt.figure()
    plt.plot(cart1.scoreEpisodes, label = "basic agent initialized randomly", color = "green")
    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score")
    plt.show()
