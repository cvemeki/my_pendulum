# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:16:48 2020

@author: tzhang
"""

from old import test
import numpy as np
import dynamicProgramming
# from deepQLearning import dqnAgentCart
from deepQLearning import dqnAgent

from QLearning import qAgent
import QLearning

if __name__ == '__main__':    
    testP1 = 2
    '''smaller final bonus range'''
    agent2 = dqnAgent("Pendulum-v0")
    agent2.terminalBonusRange = 0.05
    agent2.build()
    agent2.learning("random")