from old import test
import numpy as np
import dynamicProgramming
# from deepQLearning import dqnAgentCart
from deepQLearning import dqnAgent

from QLearning import qAgent
import QLearning

import matplotlib.pyplot as plt

import seaborn as sns

# in the main we are going to run different methodes and do the comparisons.



if __name__ == '__main__':
    testP2 = 0
    
    ''' dummy agent '''
    dummy = dqnAgent("Pendulum-v0")
    dummy.EXPLORE_RATE_INIT = 1
    dummy.EXPLORE_DECAY = 1  ## always exploring
    dummy.BATCH_SIZE = 1
    dummy.EPISODES_PER_LEARNING_MAX = 20
    dummy.build()
    dummy.learning("random")
    
    '''baseline model'''
    agent0 = dqnAgent("Pendulum-v0")
    agent0.build()
    agent0.learning("random")
    
    '''no final bonus range'''
    agent1 = dqnAgent("Pendulum-v0")
    agent1.terminalBonusRange = 0
    agent1.build()
    agent1.learning("random")
    agent1.offlineScore()
    '''smaller final bonus range'''
    agent2 = dqnAgent("Pendulum-v0")
    agent2.terminalBonusRange = 0.05
    agent2.build()
    agent2.learning("random")
        
    
    '''********** varing the reward **************'''
    '''agent3'''
    agent3 = dqnAgent("Pendulum-v0")
    agent3.rewardVelocity = 0.1
    agent3.build()
    agent3.learning("random")
    
    '''agent4'''
    agent4 = dqnAgent("Pendulum-v0")
    agent4.rewardVelocity = 0.01
    agent4.build()
    agent4.learning("random")
    
    
    '''agent5'''
    agent5 = dqnAgent("Pendulum-v0")
    agent5.rewardVelocity = 1
    agent5.build()
    agent5.learning("random")
    agent5.offlineScore()
    
    
    '''********** different action division **************'''
    
    agent6 = dqnAgent("Pendulum-v0")
#    agent6.actionSpace = np.array([-2, 2, 0])  ## binary action
    agent6.rewardVelocity = 10
    agent6.build()
    agent6.learning("random")
    agent6.offlineScore()
    
    
    '''********** evaluation part **************''' 
    '''offline evaluation'''
    angleRange = np.linspace(-np.pi, np.pi, 13, True)
    flatline = np.ones(12)*0.85
    plt.hlines(flatline, angleRange[:-1], angleRange[1:], label = "score of successful episode", color = "red", linestyle = ':')
    plt.hlines(dummy.scoreTable, angleRange[:-1], angleRange[1:], label = "dummy agent", color = "grey")
    plt.hlines(agent0.scoreTable, angleRange[:-1], angleRange[1:], label = "basic agent", color = "green")
    plt.hlines(agent1.scoreTable, angleRange[:-1], angleRange[1:], label = "agent without bonus", color = "blue")
    plt.hlines(agent2.scoreTable, angleRange[:-1], angleRange[1:], label = "agent with range = 0.05", color = "orange")
    plt.hlines(agent3.scoreTable, angleRange[:-1], angleRange[1:], label = "with velocity weight = 0.1", color = "cyan")
    plt.hlines(agent4.scoreTable, angleRange[:-1], angleRange[1:], label = "with velocity weight = 0.01", color = "magenta")
    plt.hlines(agent5.scoreTable, angleRange[:-1], angleRange[1:], label = "with velocity weight = 1", color = "black")
    plt.xlabel("initial angle range(rad)")
    plt.ylabel("average score")
    plt.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.4))
    plt.show()
    
    '''online evaluation'''
    plt.plot(dummy.scoreEpisodes, label = "dummy agent", color = "grey")
    plt.plot(agent0.scoreEpisodes, label = "basic agent", color = "green")
    plt.plot(agent1.scoreEpisodes, label = "agent without bonus", color = "blue")
    plt.plot(agent2.scoreEpisodes, label = "agent with range = 0.05", color = "orange")
    plt.plot(agent3.scoreEpisodes, label = "with velocity weight = 0.1", color = "cyan")
    plt.plot(agent4.scoreEpisodes, label = "with velocity weight = 0.01", color = "magenta")
    plt.plot(agent5.scoreEpisodes, label = "with velocity weight = 1", color = "black")

    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score")
    
    '''extractQ'''
    dummy.extractQ()
    agent0.extractQ()
    agent1.extractQ()
    agent2.extractQ()
    
    '''heatMap'''
    max_Q = np.amax(agent0.Q, axis=2)
    colormap = plt.cm.hot
    ax = sns.heatmap(max_Q, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()
    
    agent0.plotQSurface(0,1,11)
    

    
