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
    agent0.offlineScore()
    
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
    agent2.offlineScore()
        
    
    '''extractQ'''
    dummy.extractQ()
    agent0.extractQ()
    agent1.extractQ()
    agent2.extractQ()
    
    
    '''offline evaluation'''
    angleRange = np.linspace(-np.pi, np.pi, 13, True)
    flatline = np.ones(12)*0.85
    plt.hlines(flatline, angleRange[:-1], angleRange[1:], label = "score of successful episode", color = "red", linestyle = ':')
    plt.hlines(dummy.scoreTable, angleRange[:-1], angleRange[1:], label = "dummy agent", color = "grey")
    plt.hlines(agent0.scoreTable, angleRange[:-1], angleRange[1:], label = "basic agent with range = 0.1", color = "green")
    plt.hlines(agent1.scoreTable, angleRange[:-1], angleRange[1:], label = "agent without bonus", color = "blue")
    plt.hlines(agent2.scoreTable, angleRange[:-1], angleRange[1:], label = "agent with range = 0.05", color = "orange")
    plt.ylabel("average score")
    plt.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.4))
    plt.show()
    
    '''online evaluation'''
    plt.plot(dummy.scoreEpisodes, label = "dummy agent", color = "grey")
    plt.plot(agent0.scoreEpisodes, label = "basic agent with range = 0.1", color = "green")
    plt.plot(agent1.scoreEpisodes, label = "agent without bonus", color = "blue")
    plt.plot(agent2.scoreEpisodes, label = "agent with range = 0.05", color = "orange")

    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score")

    '''heatMap'''
    max_Q = np.amax(agent2.Q, axis=2)
    colormap = plt.cm.hot
    ax = sns.heatmap(max_Q, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()
    
    agent0.plotQSurface(0,1,11)
    agent0.plotQSurface(0,2,33)
    agent0.plotQSurface(1,2,33)
    
    
    '''********** varing the reward **************'''
    '''agent3'''
    agent3 = dqnAgent("Pendulum-v0")
    agent3.rewardVelocity = 0.1
    agent3.build()
    agent3.learning("random")
    agent3.offlineScore()
    
    '''agent4'''
    agent4 = dqnAgent("Pendulum-v0")  # TODO: rerun this
    agent4.rewardVelocity = 0.01
    agent4.build()
    agent4.learning("random")
    agent4.offlineScore()
    
    
    '''agent5'''
    agent5 = dqnAgent("Pendulum-v0")  # unsolved in 50 episode
    agent5.rewardVelocity = 1
    agent5.build()
    agent5.learning("random")
    agent5.offlineScore()
    
    ''' agent6'''
    agent6 = dqnAgent("Pendulum-v0")
    agent6.rewardVelocity = 10
    agent6.build()
    agent6.learning("random")
    agent6.offlineScore()
    
    '''extractQ'''
    agent3.extractQ()
    agent4.extractQ()
    agent5.extractQ()
    agent6.extractQ()
    
    '''offline evaluation'''
    angleRange = np.linspace(-np.pi, np.pi, 13, True)
    flatline = np.ones(12)*0.85
    plt.hlines(flatline, angleRange[:-1], angleRange[1:], label = "score of successful episode", color = "red", linestyle = ':')
#    plt.hlines(dummy.scoreTable, angleRange[:-1], angleRange[1:], label = "dummy agent", color = "grey")
    plt.hlines(agent0.scoreTable, angleRange[:-1], angleRange[1:], label = "basic agent with weight = 0", color = "green")
    plt.hlines(agent4.scoreTable, angleRange[:-1], angleRange[1:], label = "with velocity weight = 0.01", color = "orange")
#    plt.hlines(agent3.scoreTable, angleRange[:-1], angleRange[1:], label = "with velocity weight = 0.1", color = "blue")
    plt.hlines(agent5.scoreTable, angleRange[:-1], angleRange[1:], label = "with velocity weight = 1", color = "cyan")
#    plt.hlines(agent6.scoreTable, angleRange[:-1], angleRange[1:], label = "with velocity weight = 10", color = "grey")
    plt.xlabel("initial angle range(rad)")
    plt.ylabel("average score")
    plt.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.4))
    plt.show()
    
    '''online evaluation'''
#    plt.plot(dummy.scoreEpisodes, label = "dummy agent", color = "grey")
    plt.plot(agent0.scoreEpisodes, label = "basic agent with weight = 0", color = "green")
    plt.plot(agent4.scoreEpisodes, label = "with velocity weight = 0.01", color = "orange")
#    plt.plot(agent3.scoreEpisodes[:30], label = "with velocity weight = 0.1", color = "blue")
    plt.plot(agent5.scoreEpisodes, label = "with velocity weight = 1", color = "cyan")
#    plt.plot(agent6.scoreEpisodes[:30], label = "with velocity weight = 10", color = "grey")

    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score")
    

    '''heatMap'''
    max_Q = np.amax(agent5.Q, axis=2)
    colormap = plt.cm.hot
    ax = sns.heatmap(max_Q, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()
    
    
    '''surface plot'''
    agent6.plotQSurface(0,1,11)
    agent6.plotQSurface(0,2,33)
    agent6.plotQSurface(1,2,33)
    
    '''********** different action division **************'''
    
    agent7 = dqnAgent("Pendulum-v0")
    agent7.actionSpace = np.array([-2, 2, 0])  ## binary action
    agent7.build()
    agent7.learning("random")
    agent7.offlineScore()
    
    agent8 = dqnAgent("Pendulum-v0")
    agent8.actionSpace = np.linspace(-2, 2, 11, True)
    agent8.build()
    agent8.learning("random")
    agent8.offlineScore()
    
    agent9 = dqnAgent("Pendulum-v0")
    agent9.actionSpace = np.linspace(-2, 2, 41, True)
    agent9.build()
    agent9.learning("random")
    agent9.offlineScore()
    
    agent10 = dqnAgent("Pendulum-v0")
    agent10.actionSpace = np.linspace(-2, 2, 20, True)
    agent10.build()
    agent10.learning("random")
    agent10.offlineScore()
    
        
    '''extractQ'''
    agent7.extractQ()
    agent8.extractQ()
    agent9.extractQ()
    
    '''offline evaluation'''
    angleRange = np.linspace(-np.pi, np.pi, 13, True)
    flatline = np.ones(12)*0.85
    plt.hlines(flatline, angleRange[:-1], angleRange[1:], label = "score of successful episode", color = "red", linestyle = ':')
    plt.hlines(agent0.scoreTable, angleRange[:-1], angleRange[1:], label = "basic agent with 21 actions", color = "green")
    plt.hlines(agent7.scoreTable, angleRange[:-1], angleRange[1:], label = "agent with 3 actions", color = "blue")
    plt.hlines(agent8.scoreTable, angleRange[:-1], angleRange[1:], label = "agent with 11 actions", color = "orange")
#    plt.hlines(agent9.scoreTable, angleRange[:-1], angleRange[1:], label = "with velocity weight = 0.1", color = "grey")
#    plt.hlines(agent4.scoreTable, angleRange[:-1], angleRange[1:], label = "with velocity weight = 0.01", color = "magenta")
#    plt.hlines(agent5.scoreTable, angleRange[:-1], angleRange[1:], label = "with velocity weight = 1", color = "black")
    plt.xlabel("initial angle range(rad)")
    plt.ylabel("average score")
    plt.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.4))
    plt.show()
    
    '''online evaluation'''
    plt.plot(agent0.scoreEpisodes, label = "basic agent with 21 actions", color = "green")
    plt.plot(agent7.scoreEpisodes, label = "agent with 3 actions", color = "blue")
    plt.plot(agent8.scoreEpisodes, label = "agent with 11 actions", color = "orange")
#    plt.plot(agent9.scoreEpisodes, label = "with velocity weight = 0.1", color = "grey")
#    plt.plot(agent4.scoreEpisodes, label = "with velocity weight = 0.01", color = "magenta")
#    plt.plot(agent5.scoreEpisodes, label = "with velocity weight = 1", color = "black")

    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score")

    '''heatMap'''
    max_Q = np.amax(agent8.Q, axis=2)
    colormap = plt.cm.hot
    ax = sns.heatmap(max_Q, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()
    
    agent7.plotQSurface(0,1,1)
    agent7.plotQSurface(0,2,33)
    agent7.plotQSurface(1,2,33)
    
    '''********** different înitial states **************'''
    
    agent11 = dqnAgent("Pendulum-v0")
    agent11.build()
    agent11.learning(([np.pi,-1],[np.pi,1]))
    agent11.offlineScore()
    
    agent12 = dqnAgent("Pendulum-v0")
    agent12.build()
    agent12.learning(([0,-1],[0,1]))
    agent12.offlineScore()
    
    '''********** different learning rate **************'''
    agent13 = dqnAgent("Pendulum-v0")
    agent13.LEARNING_RATE = 1
    agent13.build()
    agent13.learning("random")
    agent13.offlineScore()
    
    agent14 = dqnAgent("Pendulum-v0")
    agent14.LEARNING_RATE = 0.5
    agent14.build()
    agent14.learning("random")
    agent14.offlineScore()
    
    '''********** different discount factor **************'''
    
    agent15 = dqnAgent("Pendulum-v0")
    agent15.GAMMA = 0.8
    agent15.build()
    agent15.learning("random")
    agent15.offlineScore()
    
    agent16 = dqnAgent("Pendulum-v0")
    agent16.GAMMA = 0.5
    agent16.build()
    agent16.learning("random")
    agent16.offlineScore()
    
    '''********** different replay **************'''
    
    agent17 = dqnAgent("Pendulum-v0")
    agent17.replay = 0
    agent17.build()
    agent17.learning("random")
    agent17.offlineScore()
    
    agent18 = dqnAgent("Pendulum-v0")
    agent18.BATCH_SIZE = 20
    agent18.build()
    agent18.learning("random")
    agent18.offlineScore()

