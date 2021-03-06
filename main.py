import numpy as np

# from deepQLearning import dqnAgentCart
from deepQLearning import dqnAgent

from QLearning import qAgent

import matplotlib.pyplot as plt

import seaborn as sns

# in the main we are going to run different methodes and do the comparisons.



if __name__ == '__main__':
    testP2 = 0
    
    ''' dummy agent '''
    dummy = dqnAgent("Pendulum-v0")
    dummy.BATCH_SIZE = 1
    dummy.EPISODES_PER_LEARNING_MAX = 2
    dummy.build()
    dummy.learning("random")
    dummy.offlineScore()
    
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
    plt.hlines(agent5.scoreTable, angleRange[:-1], angleRange[1:], label = "with velocity weight = 1", color = "blue")
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
    plt.plot(agent5.scoreEpisodes, label = "with velocity weight = 1", color = "blue")
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
    
    print()
    
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
    plt.hlines(agent9.scoreTable, angleRange[:-1], angleRange[1:], label = "agent with 41 actions", color = "brown")
    plt.xlabel("initial angle range(rad)")
    plt.ylabel("average score")
    plt.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.4))
    plt.show()
    
    '''online evaluation'''
    plt.plot(agent0.scoreEpisodes, label = "basic agent with 21 actions", color = "green")
    plt.plot(agent7.scoreEpisodes, label = "agent with 3 actions", color = "blue")
    plt.plot(agent8.scoreEpisodes, label = "agent with 11 actions", color = "orange")
    plt.plot(agent9.scoreEpisodes, label = "agent with 41 actions", color = "brown")

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
    
    agent11 = dqnAgent("Pendulum-v0")  # fail in 50 eps
    agent11.build()
    agent11.learning(([np.pi,-1],[np.pi,1]))
    agent11.offlineScore()
    
    agent12 = dqnAgent("Pendulum-v0")   # fail in 50 eps
    agent12.build()
    agent12.learning(([0,-1],[0,1]))
    agent12.offlineScore()
    
    '''extractQ'''
    agent11.extractQ()
    agent12.extractQ()
    
    '''offline evaluation'''
    angleRange = np.linspace(-np.pi, np.pi, 13, True)
    flatline = np.ones(12)*0.85
    plt.hlines(flatline, angleRange[:-1], angleRange[1:], label = "score of successful episode", color = "red", linestyle = ':')
    plt.hlines(agent0.scoreTable, angleRange[:-1], angleRange[1:], label = "basic agent initialized randomly", color = "green")
    plt.hlines(agent11.scoreTable, angleRange[:-1], angleRange[1:], label = "agent initialized always from the bottom", color = "blue")
    plt.hlines(agent12.scoreTable, angleRange[:-1], angleRange[1:], label = "agent initialized always from the top", color = "orange")
    plt.xlabel("initial angle range(rad)")
    plt.ylabel("average score")
    plt.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.4))
    plt.show()
    
    '''online evaluation'''
    plt.plot(agent0.scoreEpisodes, label = "basic agent initialized randomly", color = "green")
    plt.plot(agent11.scoreEpisodes, label = "agent initialized always from the bottom", color = "blue")
    plt.plot(agent12.scoreEpisodes, label = "agent initialized always from the top", color = "orange")

    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score")

    '''heatMap'''
    max_Q = np.amax(agent12.Q, axis=2)
    colormap = plt.cm.hot
    ax = sns.heatmap(max_Q, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()
    
    agent12.plotQSurface(0,1,11)
    agent12.plotQSurface(0,2,33)
    agent12.plotQSurface(1,2,33)
    
    
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
    
    '''extractQ'''
    agent13.extractQ()
    agent14.extractQ()
    
    '''offline evaluation'''
    angleRange = np.linspace(-np.pi, np.pi, 13, True)
    flatline = np.ones(12)*0.85
    plt.hlines(flatline, angleRange[:-1], angleRange[1:], label = "score of successful episode", color = "red", linestyle = ':')
    plt.hlines(agent0.scoreTable, angleRange[:-1], angleRange[1:], label = "basic agent with learning rate = 0.8", color = "green")
    plt.hlines(agent13.scoreTable, angleRange[:-1], angleRange[1:], label = "agent with learning rate = 1", color = "blue")
    plt.hlines(agent14.scoreTable, angleRange[:-1], angleRange[1:], label = "agent with learning rate = 0.5", color = "orange")
    plt.xlabel("initial angle range(rad)")
    plt.ylabel("average score")
    plt.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.4))
    plt.show()
    
    '''online evaluation'''
    plt.plot(agent0.scoreEpisodes, label = "basic agent with learning rate = 0.8", color = "green")
    plt.plot(agent13.scoreEpisodes, label = "agent with learning rate = 1", color = "blue")
    plt.plot(agent14.scoreEpisodes, label = "agent with learning rate = 0.5", color = "orange")

    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score")

    '''heatMap'''
    max_Q = np.amax(agent13.Q, axis=2)
    colormap = plt.cm.hot
    ax = sns.heatmap(max_Q, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()


    bestAction = np.argmax(agent14.Q, axis=2)
    bestAction = agent14.actionSpace[bestAction]
    colormap = plt.cm.hot
    ax = sns.heatmap(bestAction, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()
    
    agent13.plotQSurface(0,1,11)
    agent13.plotQSurface(0,2,33)
    agent13.plotQSurface(1,2,33)

    
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
    
    '''extractQ'''
    agent15.extractQ()
    agent16.extractQ()
    
    '''offline evaluation'''
    angleRange = np.linspace(-np.pi, np.pi, 13, True)
    flatline = np.ones(12)*0.85
    plt.hlines(flatline, angleRange[:-1], angleRange[1:], label = "score of successful episode", color = "red", linestyle = ':')
    plt.hlines(agent0.scoreTable, angleRange[:-1], angleRange[1:], label = "basic agent with discount factor = 0.95", color = "green")
    plt.hlines(agent15.scoreTable, angleRange[:-1], angleRange[1:], label = "agent with discount factor = 0.8", color = "blue")
    plt.hlines(agent16.scoreTable, angleRange[:-1], angleRange[1:], label = "agent with discount factor = 0.5", color = "orange")
    plt.xlabel("initial angle range(rad)")
    plt.ylabel("average score")
    plt.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.4))
    plt.show()
    
    '''online evaluation'''
    plt.plot(agent0.scoreEpisodes, label = "basic agent with discount factor = 0.95", color = "green")
    plt.plot(agent15.scoreEpisodes, label = "agent with discount factor = 0.8", color = "blue")
    plt.plot(agent16.scoreEpisodes, label = "agent with discount factor = 0.5", color = "orange")

    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score")

    '''heatMap'''
    max_Q = np.amax(agent16.Q, axis=2)
    colormap = plt.cm.hot
    ax = sns.heatmap(max_Q, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()


    bestAction = np.argmax(agent15.Q, axis=2)
    bestAction = agent15.actionSpace[bestAction]
    colormap = plt.cm.hot
    ax = sns.heatmap(bestAction, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()
    
    agent16.plotQSurface(0,1,11)
    agent16.plotQSurface(0,2,33)
    agent16.plotQSurface(1,2,33)
    
    '''********** different replay **************'''    
    agent171 = dqnAgent("Pendulum-v0")
    agent171.replay = 0
    agent171.EPISODES_PER_LEARNING_MAX = 2000
    agent171.EXPLORE_DECAY = 0.99999
    agent171.build()
    agent171.learning("random")
    agent171.offlineScore()
    
    agent172 = dqnAgent("Pendulum-v0")
    agent172.replay = 1
    agent172.BATCH_SIZE = 1
    agent172.EPISODES_PER_LEARNING_MAX = 2000
    agent172.EXPLORE_DECAY = 0.99999
    agent172.build()
    agent172.learning("random")
    agent172.offlineScore()
    
    agent18 = dqnAgent("Pendulum-v0")
    agent18.BATCH_SIZE = 20
    agent18.EPISODES_PER_LEARNING_MAX = 500
    agent18.build()
    agent18.learning("random")
    agent18.offlineScore()
    
    '''extractQ'''
    agent171.extractQ()
    agent172.extractQ()
    dummy.extractQ()
    agent18.extractQ()
    
    '''offline evaluation'''
    angleRange = np.linspace(-np.pi, np.pi, 13, True)
    flatline = np.ones(12)*0.85
    plt.hlines(flatline, angleRange[:-1], angleRange[1:], label = "score of successful episode", color = "red", linestyle = ':')
#    plt.hlines(dummy.scoreTable, angleRange[:-1], angleRange[1:], label = "dummy agent", color = "grey")
    plt.hlines(agent0.scoreTable, angleRange[:-1], angleRange[1:], label = "basic agent with replay, batch size = 200, exlore decay = 0.998", color = "green")
    plt.hlines(agent171.scoreTable, angleRange[:-1], angleRange[1:], label = "agent without replay, explore decay = 0.99999", color = "blue")
    plt.hlines(agent172.scoreTable, angleRange[:-1], angleRange[1:], label = "agent with replay, batch size = 1, explore decay = 0.99999", color = "orange")
    plt.hlines(agent18.scoreTable, angleRange[:-1], angleRange[1:], label = "agent with replay, batch size = 20, exlore decay = 0.998", color = "brown")
    plt.xlabel("initial angle range(rad)")
    plt.ylabel("average score")
    plt.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.4))
    plt.show()
    
    '''online evaluation'''
    plt.plot(agent0.scoreEpisodes, label = "basic agent with replay, batch size = 200, explore decay = 0.998", color = "green")
    plt.plot(agent171.scoreEpisodes[:1150], label = "agent without replay, explore decay = 0.99999", color = "blue")
    plt.plot(agent172.scoreEpisodes, label = "agent with replay, batch size = 1, explore decay = 0.99999", color = "orange")
    plt.plot(agent18.scoreEpisodes, label = "agent with replay, batch size = 20, exlore decay = 0.998", color = "brown")
    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score")

    '''heatMap'''
    max_Q = np.amax(agent18.Q, axis=2)
    colormap = plt.cm.hot
    ax = sns.heatmap(max_Q, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()
    
    agent171.plotQSurface(0,1,11)
    agent171.plotQSurface(0,2,33)
    agent171.plotQSurface(1,2,33)

    '''*********************************Q learning part**************************************************'''
    agent19 = qAgent("Pendulum-v0")
    agent19.EXPLORE_DECAY = 0.99999
    agent19.EPISODES_PER_LEARNING_MAX = 2000
    agent19.build()
    agent19.learning("random")
    agent19.offlineScore()
    
        
    '''offline evaluation'''
    angleRange = np.linspace(-np.pi, np.pi, 13, True)
    flatline = np.ones(12)*0.85
    plt.hlines(flatline, angleRange[:-1], angleRange[1:], label = "score of successful episode", color = "red", linestyle = ':')
    plt.hlines(dummy.scoreTable, angleRange[:-1], angleRange[1:], label = "dummy agent", color = "grey")
#    plt.hlines(agent0.scoreTable, angleRange[:-1], angleRange[1:], label = "basic agent with replay, batch size = 200, exlore decay = 0.998", color = "green")
    plt.hlines(agent19.scoreTable, angleRange[:-1], angleRange[1:], label = "basic agent q learning", color = "blue")
#    plt.hlines(agent172.scoreTable, angleRange[:-1], angleRange[1:], label = "agent with replay, batch size = 1, explore decay = 0.99999", color = "orange")
#    plt.hlines(agent18.scoreTable, angleRange[:-1], angleRange[1:], label = "agent with replay, batch size = 20, exlore decay = 0.998", color = "brown")
    plt.xlabel("initial angle range(rad)")
    plt.ylabel("average score")
    plt.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.4))
    plt.show()
    
    '''online evaluation'''
    ax = plt.axes()
#    plt.plot(agent0.scoreEpisodes, label = "basic agent with replay, batch size = 200, explore decay = 0.998", color = "green")
    plt.plot(agent19.scoreEpisodes[:1150], label = "basic agent q learning", color = "blue")
#    plt.plot(agent172.scoreEpisodes, label = "agent with replay, batch size = 1, explore decay = 0.99999", color = "orange")
#    plt.plot(agent18.scoreEpisodes, label = "agent with replay, batch size = 20, exlore decay = 0.998", color = "brown")
    ax.axhline(y=0.975, linestyle = ':', color = 'r', label = "score considered as solved")
    plt.legend()
    plt.xlabel("episodes")
    plt.ylabel("episode score")

    '''heatMap'''
    max_Q = np.amax(agent19.Q, axis=2)
    colormap = plt.cm.hot
    ax = sns.heatmap(max_Q, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()

    agent19.plotQSurface(0,1,11)
    agent19.plotQSurface(0,2,33)
    agent19.plotQSurface(1,2,33)
    
    '''*********************************Reduced Q learning part**************************************************'''
    
    agent20 = qAgent("Pendulum-v0")
    agent20.EXPLORE_DECAY = 0.9998
    agent20.stateSpace = (np.linspace(-np.pi/3, np.pi/3, 33, False), np.linspace(-4, 4, 33, False))
    agent20.state_bound = ([-np.pi/6,-1],[np.pi/6,1])
    agent20.EPISODES_PER_LEARNING_MAX = 2000
    agent20.build()
    agent20.learning(([-np.pi/3,-4],[np.pi/3,4]))
    agent20.offlineScore()
    
