# Represents the Q-learning agent. No deep network, just Q table iteration.
import gym
import numpy as np
import helper
import seaborn as sns
sns.set()
import random
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import gym_maze

class qAgent:

    def __init__(self, envName):
    
        '''changeable parameters'''
        
        self.MEMORY_SIZE = 100000
        self.scoreStep = 0
        self.scoreEpisode = 0

        self.scoreEpisodes = []
        self.scoreSteps = []

        self.testScore = 0

        '''environment'''
        self.env = gym.make(envName)
        self.envName = envName

        self.rewardVelocity = 0
        
        if self.envName == "Pendulum-v0":   ## TODO: to check
            self.actionSpace = np.linspace(-2, 2, 21, True)
            # self.actionSpace = np.array([-2, 2, 0])  # what if binary action(bang-bang control)?
            self.stateSpace = (np.linspace(-np.pi, np.pi, 65, False), np.linspace(-8, 8, 65, False)) # it's a tuple
            
            
        if self.envName == "CartPole-v0":
            self.actionSpace = np.array([0,1])  # what if binary action(bang-bang control)?
            
        if "maze" in self.envName:
            self.actionSpace = np.arange(4)  # what if binary action(bang-bang control)?
            self.state_bound = list(zip(self.env.observation_space.low, self.env.observation_space.high))
            self.stateSpace = (np.arange(self.state_bound[0][1]+1), np.arange(self.state_bound[1][1]+1))
            

        '''learning'''
        if self.envName == "Pendulum-v0":  # baseline model parameters   ## TODO: to check
            self.BATCH_SIZE = 200
            self.GAMMA = 0.95
            self.LEARNING_RATE = 0.8
            self.STEPS_PER_EPISODE_MAX = 200
            self.EPISODES_PER_LEARNING_MAX = 50
            self.CONSECUTIVE_EPISODE = 5
            self.terminalBonus = 100
            self.terminalBonusRange = 0.1
            self.SCORE_SOLVED = 0.85   ## max = 1

        if self.envName == "CartPole-v0":  # baseline model parameters
            self.BATCH_SIZE = 200
            self.GAMMA = 0.99
            self.LEARNING_RATE = 0.99
            self.STEPS_PER_EPISODE_MAX = 200
            self.EPISODES_PER_LEARNING_MAX = 20
            self.CONSECUTIVE_EPISODE = 5

            self.SCORE_SOLVED = 195/200  ## max = 1

        if "maze" in self.envName:
            self.BATCH_SIZE = 200
            self.GAMMA = 0.99
            self.LEARNING_RATE = 0.99
            self.STEPS_PER_EPISODE_MAX = 200
            self.EPISODES_PER_LEARNING_MAX = 200
            self.CONSECUTIVE_EPISODE = 5

            self.SCORE_SOLVED = 195/200  ## max = 1
            
            
        '''exploring'''
        self.EXPLORE_RATE_INIT = 1.0
        self.EXPLORE_MAX = 1.0
        self.EXPLORE_MIN = 0.01
        self.EXPLORE_DECAY = 0.998

        '''testing'''
        self.TEST_STEP = 1000

        '''just initialization'''

        self.render = 0
        if self.envName == "CartPole-v0":
            self.render = 1
        '''learning parameters'''
        self.nLearningStep = 0
        self.nLearningEpisode = 0
        self.nSuccessfulEpisode = 0
        self.nTestingStep = 0
        self.terminateEpisode = 0
        self.terminateLearning = 0

        '''internal'''
        self.reward = 0

        print("Milestone: initialized finished")

        '''build'''

    def build(self):
        self.memory = deque(maxlen=self.MEMORY_SIZE) # default 1000000
        
        self.EXPLORE_RATE = self.EXPLORE_RATE_INIT
        self.action = self.actionSpace[0]
        self.stateSpace1 = self.stateSpace[0]
        self.stateSpace2 = self.stateSpace[1]
        self.state = [[self.stateSpace1[0], self.stateSpace2[0]]] # a tuple
        self.oldState = self.state

        self.observationDim = self.env.observation_space.shape[0]  # = 3
        self.actionNumber = self.actionSpace.shape[0]
        self.Q = np.zeros((self.stateSpace1.size, self.stateSpace2.size, self.actionSpace.size))
        
        self.env.seed(2019)  # to compare

        
    '''Internal mechanisms'''
    def remember(self, state, action, reward, state_next, terminal):
        self.memory.append((state, action, reward, state_next, terminal))

    def experienceReplay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = random.sample(self.memory, self.BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            best_q = np.amax(self.Q[state_next])
            self.Q[self.oldState + (action,)] = (1 - self.LEARNING_RATE) * self.Q[state + (action,)] + self.LEARNING_RATE * \
                                              (self.reward + self.GAMMA * best_q)

    '''make actions'''
    def act(self):
        self.action = np.argmax(self.Q[self.state])

    def explore(self):
        if np.random.rand() < self.EXPLORE_RATE:
            self.action = random.randrange(self.actionNumber)
            print("maze action ", self.action)
            return

        if self.envName == "Pendulum-v0":   ## TODO: to check
            maxID = np.argwhere(self.Q[self.state] == np.amax(self.Q[self.state]))
            self.action = random.choice(maxID)
            # self.action = np.argmax(self.Q[self.state])  

        if self.envName == "CartPole-v0":
            maxID = np.argwhere(self.Q[self.state] == np.amax(self.Q[self.state]))
            self.action = random.choice(maxID)
            # self.action = np.argmax(self.Q[self.state])  
            
        if "maze" in self.envName:
            maxID = np.argwhere(self.Q[self.state] == np.amax(self.Q[self.state]))
            self.action = int(random.choice(maxID[:,0]))
            # self.action = np.argmax(self.Q[self.state])  
            
    '''Interact with environment'''
    def obvToState(self, obv):  # interface with the environment
        if self.envName == "Pendulum-v0":  ## TODO: to check
            self.state = np.reshape(obv, [1, self.observationDim])
        
        if self.envName == "CartPole-v0":
            return
            
        if "maze" in self.envName:
            self.state = (int(obv[0]), int(obv[1])) 
            
            
    def actionToImpact(self):  # interface with the environment
        if self.envName == "Pendulum-v0": ## TODO: to check
            return tuple([self.actionSpace[self.action],])
        if self.envName == "CartPole-v0":
            return self.actionSpace[self.action]
        if "maze" in self.envName:
            return self.action 

    '''learning'''
    def oneLearningStep(self):

        '''act'''
        self.explore()
        impact = self.actionToImpact()
        self.oldState = self.state

        '''observe''' 
        obv, self.reward, self.terminateEpisode, info = self.env.step(impact)
        self.obvToState(obv)  # state = state_next
        self.episodeTerminateCondition()
        
        if self.envName == "CartPole-v0":  ## TODO: to check
            self.reward = self.reward if not self.terminateEpisode else -self.reward

        if self.envName == "Pendulum-v0":
            self.reward = -((helper.trigo2angle(self.oldState[0][0], self.oldState[0][1]))**2 + self.rewardVelocity*self.oldState[0][2]**2 + .001*(impact[0]**2))
            if np.linalg.norm([helper.trigo2angle(obv[0], obv[1])]) < self.terminalBonusRange:
                self.reward = self.reward + self.terminalBonus

        '''update model'''
        self.remember(self.oldState, self.action, self.reward, self.state, self.terminateEpisode)
        self.experienceReplay()
        

    def oneLearningEpisode(self, mode):

        '''initialize'''
        if self.envName == "Pendulum-v0":  ## TODO: to check
            obv = self.env.reset(mode=mode)
        else:
            obv = self.env.reset()
        self.obvToState(obv)
        self.initializePerEpisode()

        '''run several steps'''
        while not self.terminateEpisode:
            self.oneLearningStep()

            '''miscellaneous update'''
            self.updatePerStep()

    def learning(self, mode):

        '''''''initialize'''
        self.initializePerLearning()

        '''run several episodes'''
        while not self.terminateLearning:
            self.initializePerEpisode()

            self.oneLearningEpisode(mode)

            '''miscellaneous update'''
            self.updatePerEpisode()

            self.learningTerminateCondition()
        self.env.close()

    '''terminate conditions'''
    def episodeTerminateCondition(self):
        if self.envName == "Pendulum-v0":  ## TODO: check
            if self.nLearningStep >= self.STEPS_PER_EPISODE_MAX:
                self.terminateEpisode = 1
            else:
                self.terminateEpisode = 0
                
        if "maze" in self.envName:
            if self.nLearningStep >= self.STEPS_PER_EPISODE_MAX:
                self.terminateEpisode = 1
            else:
                self.terminateEpisode = 0
        return None

    def learningTerminateCondition(self):
        if self.envName == "Pendulum-v0":  ## TODO: to check
            if self.nLearningEpisode >= self.EPISODES_PER_LEARNING_MAX:
                self.terminateLearning = 1
            elif len(self.scoreEpisodes) >= self.CONSECUTIVE_EPISODE:
                if np.mean(self.scoreEpisodes[-self.CONSECUTIVE_EPISODE:]) >= self.SCORE_SOLVED:
                    self.terminateLearning = 1

        if self.envName == "CartPole-v0":
            if self.nLearningEpisode >= self.EPISODES_PER_LEARNING_MAX:
                self.terminateLearning = 1
            elif len(self.scoreEpisodes) >= self.CONSECUTIVE_EPISODE:
                if np.mean(self.scoreEpisodes[-self.CONSECUTIVE_EPISODE:]) >= self.SCORE_SOLVED:
                    self.terminateLearning = 1
                    
        if "maze" in self.envName:
            if self.nLearningEpisode >= self.EPISODES_PER_LEARNING_MAX:
                self.terminateLearning = 1
                
        return None

    '''update per'''
    def updatePerStep(self):
        print("learning: Episode = ", self.nLearningEpisode, "; Step = ", self.nLearningStep, "; explore rate = ", self.EXPLORE_RATE)
        print(self.state)

        if self.render:
            self.env.render()

        if self.envName == "Pendulum-v0":   ## TODO: to check
            self.EXPLORE_RATE *= self.EXPLORE_DECAY
            self.EXPLORE_RATE = max(self.EXPLORE_MIN, self.EXPLORE_RATE)
            self.scoreStep = 1*(np.pi - np.abs(helper.trigo2angle(self.state[0][0],self.state[0][1])))/np.pi
            self.scoreEpisode = self.scoreEpisode + self.scoreStep
        if self.envName == "CartPole-v0":
            self.scoreStep = self.reward
            self.scoreEpisode = self.scoreEpisode + self.scoreStep
        if "maze" in self.envName:
            self.EXPLORE_RATE *= self.EXPLORE_DECAY
            self.EXPLORE_RATE = max(self.EXPLORE_MIN, self.EXPLORE_RATE)
            
        self.scoreSteps.append(self.scoreStep)
        # if self.envName == "Pendulum-v0":
        #     self.interactivePlotScore()

        self.nLearningStep += 1
        return None

    def updatePerEpisode(self):
        print("one episode ends")
        self.nLearningEpisode += 1
        # if self.nLearningStep >= self.STEPS_PER_EPISODE_WHEN_SUCCESS:
        #     self.nSuccessfulEpisode += 1
        if self.envName == "Pendulum-v0":
            self.scoreEpisodes.append(self.scoreEpisode / self.nLearningStep)
            self.interactivePlotScore()
            
        if self.envName == "CartPole-v0":  ## TODO: to check
            self.EXPLORE_RATE *= self.EXPLORE_DECAY
            self.EXPLORE_RATE = max(self.EXPLORE_MIN, self.EXPLORE_RATE)
            self.scoreEpisodes.append(self.scoreEpisode / 200)
            self.interactivePlotScore()
            
        if "maze" in self.envName:
            print("")
        return None

    '''initialize per'''
    def initializePerEpisode(self):
        # self.EXPLORE_RATE = self.EXPLORE_RATE_INIT/(self.nLearningEpisode+1)
        self.nLearningStep = 0
        self.terminateEpisode = 0
        self.scoreEpisode = 0
        return None

    def initializePerLearning(self):
        self.terminateLearning = 0
        self.nLearningEpisode = 0
        self.scoreSteps = []
        self.scoreEpisodes = []
        return None

    '''testing'''
    def testing(self, mode):
        if self.envName == "Pendulum-v0":   ## TODO: to check
            obv = self.env.reset(mode = mode)
        else: obv = self.env.reset()
        self.testScore = 0
        terminate = 0
        step = 0
        while not terminate:
        # for step in range(self.TEST_STEP):
            self.obvToState(obv)
            self.act()  # do an action based on current state
            impact = self.actionToImpact()
            obv, reward, terminate, _ = self.env.step(impact)
            if self.render:
                self.env.render()
            if self.envName == "Pendulum-v0":
                terminate = 0
                if step >= self.TEST_STEP:
                    self.testScore = self.testScore / self.TEST_STEP
                    break
                self.testScore = self.testScore + 1*(np.pi - np.abs(helper.trigo2angle(obv[0],obv[1])))/np.pi
            if self.envName == "CartPole-v0":
                if terminate == 1:
                    break
                self.testScore = self.testScore + reward/200
            step += 1
        self.env.close()


    '''plotting'''
    def plotQSurface(self, xDim, yDim, anchor):
        if {xDim, yDim} == {0,1}:
            slicedQ = np.reshape(self.Q[:, :, anchor], [self.Q.shape[0], self.Q.shape[1]])
            xAxis, yAxis = np.meshgrid(self.stateSpace1, self.stateSpace2)
            xLabel, yLabel = 'theta', 'dtheta'

        elif {xDim, yDim} == {0,2}:
            slicedQ = np.reshape(self.Q[:, anchor, :], [self.Q.shape[0], self.Q.shape[2]])
            xAxis, yAxis = np.meshgrid(self.stateSpace1, self.actionSpace)
            xLabel, yLabel = 'theta', 'action'

        else: # {xDim, yDim} == {1,2}
            slicedQ = np.reshape(self.Q[anchor, :, :], [self.Q.shape[1], self.Q.shape[2]])
            xAxis, yAxis = np.meshgrid(self.stateSpace2, self.actionSpace)
            xLabel, yLabel = 'dtheta', 'action'

        fig1 = plt.figure()
        ax = fig1.gca(projection='3d')
        ax.plot_surface(xAxis.T, yAxis.T, slicedQ)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_zlabel('Q value')
        plt.show()

    def interactivePlot(self):
        max_Q = np.amax(self.Q, axis=2)
        plt.clf()
        colormap = plt.cm.hot
        ax = sns.heatmap(max_Q, cmap=colormap)
        # rect = patches.Rectangle((self.state[1], self.state[0]), 1, 1, linewidth=1, edgecolor='lime', facecolor='lime')
        # ax.add_patch(rect)
        plt.show()
        plt.pause(0.01)

    def interactivePlotScore(self):
        plt.clf()
        if self.envName == "Pendulum-v0":   ## TODO: to check
            plt.plot(self.scoreEpisodes)
            plt.xlabel("episodes")
            plt.ylabel("episode score")
        if self.envName == "CartPole-v0":
            plt.plot(self.scoreEpisodes)
            plt.xlabel("episodes")
            plt.ylabel("episode score")
        plt.legend(self.envName)
        plt.show()
        plt.pause(0.01)

    def offlineScore(self):
        self.render = 0
        if self.envName == "Pendulum-v0":   ## TODO: to check
            self.TEST_STEP = 500
            angleRange = np.linspace(-np.pi, np.pi, 13, True)
            # speedRange = np.array([-1,1])
            self.scoreTable = np.zeros(len(angleRange)-1)
            for i in range(len(angleRange)-1):
                print(i)
                for step in range(10):
                    self.testing(([angleRange[i], -1],[angleRange[i+1], 1]))
                    self.scoreTable[i] = self.scoreTable[i] + self.testScore
                self.scoreTable[i] = self.scoreTable[i]/10
            plt.hlines(self.scoreTable, angleRange[:-1], angleRange[1:])
            plt.xlabel("initial angle range(rad)")
            plt.ylabel("average score")
            # plt.step(angleRange[:-1], self.scoreTable, where='post', label='post')
            # plt.plot(angleRange[:-1], self.scoreTable, 'C2o', alpha=0.5)
            plt.show()
        if self.envName == "CartPole-v0":
            nTests = 1000
            self.scoreTable = np.zeros(nTests)
            for i in range(nTests):
                print(i)
                self.testing("random")
                self.scoreTable[i] = self.testScore
            ax = sns.distplot(self.scoreTable, axlabel="test score")
            ax.set_ylabel("number of tests")
            
        


def main():
    # myDqnAgent.testing()
    # myDqnAgent.extractQ()
    # myDqnAgent.plotQSurface(0, 1, 16)
    return None


if __name__ == '__main__':
    main()