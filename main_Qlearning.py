# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:19:41 2020

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
    
    '''baseline maze model'''
    maze0 = qAgent("maze-random-10x10-plus-v0")
    maze0.build()
    maze0.learning("random")
    maze0.offlineScore()