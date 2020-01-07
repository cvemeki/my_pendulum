# Represents the DQN agent
import gym
import numpy as np
import helper
import seaborn as sns
sns.set()
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
import keras
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

## for cartpole environment
class dqnAgentCart:

    def __init__(self):
        self.memory = deque(maxlen=1000000)
        '''changeable parameters'''

        '''environment'''
        self.env = gym.make("CartPole-v1")
        # self.actionSpace = np.linspace(-2, 2, 21, True)
        # self.actionSpace = np.array([-1,1,0])  # what if binary action(bang-bang control)?
        self.actionSpace = np.array([0,1])
        self.stateSpace = (np.linspace(-np.pi, np.pi, 65, False), np.linspace(-8, 8, 65, False)) # it's a tuple
        self.stateSpace1 = self.stateSpace[0]
        self.stateSpace2 = self.stateSpace[1]

        '''constants'''

        '''neural network model'''

        '''learning'''
        self.BATCH_SIZE = 20
        self.GAMMA = 0.95
        self.STEPS_PER_EPISODE_MAX = 200
        self.STEPS_PER_EPISODE_WHEN_SUCCESS = 50

        '''exploring'''
        self.EXPLORE_RATE = 0.5
        self.EXPLORE_MAX = 1.0
        self.EXPLORE_MIN = 0.01
        self.EXPLORE_DECAY = 0.995

        '''testing'''
        self.TEST_STEP = 1000

        '''just initialization'''

        self.render = 0
        '''learning parameters'''
        self.nLearningStep = 0
        self.nLearningEpisode = 0
        self.nSuccessfulEpisode = 0
        self.nTestingStep = 0
        self.terminateEpisode = 0
        self.terminateLearning = 0

        '''internal'''
        self.action = self.actionSpace[0]
        self.state = (self.stateSpace1[0], self.stateSpace2[0])  # a tuple

        '''model'''
        self.model = Sequential()
        self.observationDim = self.env.observation_space.shape[0]  # = 3
        self.actionNumber = self.actionSpace.shape[0]
        self.Q = np.zeros((self.stateSpace1.size, self.stateSpace2.size, self.actionSpace.size))
        print("Milestone: initialized finished")

        '''build'''
        self.model.add(Dense(24, input_shape=(self.observationDim,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.actionNumber, activation="linear"))
        # self.model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True))
        self.model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.001))

        print("Milestone: build finished")
        self.model.summary()

    '''Internal mechanisms'''
    def remember(self, state, action, reward, state_next, terminal):
        self.memory.append((state, action, reward, state_next, terminal))

    def experienceReplay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = random.sample(self.memory, self.BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update  # action is the index too
            self.model.fit(state, q_values, verbose=0)  # the memory before was kept too!

    '''make actions'''
    def act(self):
        q_values = self.model.predict(self.state)
        self.action = np.argmax(q_values[0])

    def explore(self):
        if np.random.rand() < self.EXPLORE_RATE:
            self.action = random.randrange(self.actionNumber)
            return
        q_values = self.model.predict(self.state)
        self.action = np.argmax(q_values[0])  # best action index at current state

    '''Interact with environment'''
    def obvToState(self, obv):  # interface with the environment
        self.state = np.reshape(obv, [1, self.observationDim])

    def actionToImpact(self):  # interface with the environment
        # impact = (self.actionSpace[self.action], )
        impact = self.actionSpace[self.action]
        return impact

    '''learning'''
    def oneLearningStep(self):
        self.nLearningStep += 1
        # print("learning: Episode = ", self.nLearningEpisode, "; Step = ", self.nLearningStep, "; explore rate = ", self.EXPLORE_RATE)
        # print(self.state)
        self.explore()
        impact = self.actionToImpact()
        oldState = self.state
        obv, reward, self.terminateEpisode, info = self.env.step(impact)
        self.obvToState(obv)  # state = state_next
        reward = reward if not self.terminateEpisode else -reward
        # if self.nLearningStep >= self.STEPS_PER_EPISODE_MAX:
        #     self.terminateEpisode = 1
        # reward and terminate condition
        # if np.linalg.norm([helper.trigo2angle(obv[0], obv[1]), obv[2]]) < 0.1:
        #     self.terminateEpisode = 1
        #     reward = reward + 100
        self.remember(oldState, self.action, reward, self.state, self.terminateEpisode)
        self.experienceReplay()
        if self.render:
            self.env.render()
        self.EXPLORE_RATE *= self.EXPLORE_DECAY
        self.EXPLORE_RATE = max(self.EXPLORE_MIN, self.EXPLORE_RATE)

    def oneLearningEpisode(self, mode):
        self.nLearningEpisode += 1
        # obv = self.env.reset(mode=mode)
        obv = self.env.reset()
        self.obvToState(obv)
        self.nLearningStep = 0
        self.terminateEpisode = 0
        while not self.terminateEpisode:
            self.oneLearningStep()
        # print("one episode ends")
        # if self.nLearningStep >= self.STEPS_PER_EPISODE_WHEN_SUCCESS:
        #     self.nSuccessfulEpisode += 1

    def learning(self, mode):
        self.nLearningEpisode = 0
        # while self.nSuccessfulEpisode < 100:
        while self.nLearningEpisode < 100:
            print("episode = ", self.nLearningEpisode, " Step = ", self.nLearningStep)
            # self.EXPLORE_RATE = 0.5  # reset the learning rate for every episode. (otherwise it's too low for the following episode)
            self.oneLearningEpisode(mode)
        self.env.close()

    '''testing'''
    def testing(self, mode):
        obv = self.env.reset(mode = mode)
        # obv = self.env.reset()
        for step in range(self.TEST_STEP):
            self.obvToState(obv)
            self.act()  # do an action based on current state
            impact = self.actionToImpact()
            obv, _, _, _ = self.env.step(impact)
            if self.render:
                self.env.render()
        self.env.close()

    '''extract the Q matrix from the model'''
    def extractQ(self):
        for i, theta in enumerate(self.stateSpace1):
            for j, dtheta in enumerate(self.stateSpace2):
                state = np.array([np.cos(theta), np.sin(theta), dtheta])
                state = np.reshape(state, [1, self.observationDim])
                q_values = self.model.predict(state)
                self.Q[i, j, :] = q_values[0, :]

    # def extractR(self):

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
        # ********* interactive plot ********************* #
        max_Q = np.amax(self.Q, axis=2)
        plt.clf()
        colormap = plt.cm.hot
        # changingData = np.random.rand(10, 12)
        ax = sns.heatmap(max_Q, cmap=colormap)
        rect = patches.Rectangle((self.state[1], self.state[0]), 1, 1, linewidth=1, edgecolor='lime', facecolor='lime')
        ax.add_patch(rect)
        plt.show()
        plt.pause(0.01)
        # ********* interactive plot ********************* #


## for pendulum environment
class dqnAgentPend:

    def __init__(self):
        self.memory = deque(maxlen=1000000)
        '''changeable parameters'''

        '''environment'''
        self.env = gym.make("CartPole-v1")
        # self.actionSpace = np.linspace(-2, 2, 21, True)
        # self.actionSpace = np.array([-1,1,0])  # what if binary action(bang-bang control)?
        self.actionSpace = np.array([0,1])
        self.stateSpace = (np.linspace(-np.pi, np.pi, 65, False), np.linspace(-8, 8, 65, False)) # it's a tuple
        self.stateSpace1 = self.stateSpace[0]
        self.stateSpace2 = self.stateSpace[1]

        '''constants'''

        '''neural network model'''

        '''learning'''
        self.BATCH_SIZE = 20
        self.GAMMA = 0.95
        self.STEPS_PER_EPISODE_MAX = 200
        self.STEPS_PER_EPISODE_WHEN_SUCCESS = 50

        '''exploring'''
        self.EXPLORE_RATE = 0.5
        self.EXPLORE_MAX = 1.0
        self.EXPLORE_MIN = 0.01
        self.EXPLORE_DECAY = 0.995

        '''testing'''
        self.TEST_STEP = 1000

        '''just initialization'''

        self.render = 0
        '''learning parameters'''
        self.nLearningStep = 0
        self.nLearningEpisode = 0
        self.nSuccessfulEpisode = 0
        self.nTestingStep = 0
        self.terminateEpisode = 0
        self.terminateLearning = 0

        '''internal'''
        self.action = self.actionSpace[0]
        self.state = (self.stateSpace1[0], self.stateSpace2[0])  # a tuple

        '''model'''
        self.model = Sequential()
        self.observationDim = self.env.observation_space.shape[0]  # = 3
        self.actionNumber = self.actionSpace.shape[0]
        self.Q = np.zeros((self.stateSpace1.size, self.stateSpace2.size, self.actionSpace.size))
        print("Milestone: initialized finished")

        '''build'''
        self.model.add(Dense(24, input_shape=(self.observationDim,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.actionNumber, activation="linear"))
        # self.model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True))
        self.model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.001))

        print("Milestone: build finished")
        self.model.summary()

    '''Internal mechanisms'''
    def remember(self, state, action, reward, state_next, terminal):
        self.memory.append((state, action, reward, state_next, terminal))

    def experienceReplay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = random.sample(self.memory, self.BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update  # action is the index too
            self.model.fit(state, q_values, verbose=0)  # the memory before was kept too!

    '''make actions'''
    def act(self):
        q_values = self.model.predict(self.state)
        self.action = np.argmax(q_values[0])

    def explore(self):
        if np.random.rand() < self.EXPLORE_RATE:
            self.action = random.randrange(self.actionNumber)
            return
        q_values = self.model.predict(self.state)
        self.action = np.argmax(q_values[0])  # best action index at current state

    '''Interact with environment'''
    def obvToState(self, obv):  # interface with the environment
        self.state = np.reshape(obv, [1, self.observationDim])

    def actionToImpact(self):  # interface with the environment
        # impact = (self.actionSpace[self.action], )
        impact = self.actionSpace[self.action]
        return impact

    '''learning'''
    def oneLearningStep(self):
        self.nLearningStep += 1
        # print("learning: Episode = ", self.nLearningEpisode, "; Step = ", self.nLearningStep, "; explore rate = ", self.EXPLORE_RATE)
        # print(self.state)
        self.explore()
        impact = self.actionToImpact()
        oldState = self.state
        obv, reward, self.terminateEpisode, info = self.env.step(impact)
        self.obvToState(obv)  # state = state_next
        reward = reward if not self.terminateEpisode else -reward
        # if self.nLearningStep >= self.STEPS_PER_EPISODE_MAX:
        #     self.terminateEpisode = 1
        # reward and terminate condition
        # if np.linalg.norm([helper.trigo2angle(obv[0], obv[1]), obv[2]]) < 0.1:
        #     self.terminateEpisode = 1
        #     reward = reward + 100
        self.remember(oldState, self.action, reward, self.state, self.terminateEpisode)
        self.experienceReplay()
        if self.render:
            self.env.render()
        self.EXPLORE_RATE *= self.EXPLORE_DECAY
        self.EXPLORE_RATE = max(self.EXPLORE_MIN, self.EXPLORE_RATE)

    def oneLearningEpisode(self, mode):
        self.nLearningEpisode += 1
        # obv = self.env.reset(mode=mode)
        obv = self.env.reset()
        self.obvToState(obv)
        self.nLearningStep = 0
        self.terminateEpisode = 0
        while not self.terminateEpisode:
            self.oneLearningStep()
        # print("one episode ends")
        # if self.nLearningStep >= self.STEPS_PER_EPISODE_WHEN_SUCCESS:
        #     self.nSuccessfulEpisode += 1

    def learning(self, mode):
        self.nLearningEpisode = 0
        # while self.nSuccessfulEpisode < 100:
        while self.nLearningEpisode < 100:
            print("episode = ", self.nLearningEpisode, " Step = ", self.nLearningStep)
            # self.EXPLORE_RATE = 0.5  # reset the learning rate for every episode. (otherwise it's too low for the following episode)
            self.oneLearningEpisode(mode)
        self.env.close()

    '''testing'''
    def testing(self, mode):
        obv = self.env.reset(mode = mode)
        # obv = self.env.reset()
        for step in range(self.TEST_STEP):
            self.obvToState(obv)
            self.act()  # do an action based on current state
            impact = self.actionToImpact()
            obv, _, _, _ = self.env.step(impact)
            if self.render:
                self.env.render()
        self.env.close()

    '''extract the Q matrix from the model'''
    def extractQ(self):
        for i, theta in enumerate(self.stateSpace1):
            for j, dtheta in enumerate(self.stateSpace2):
                state = np.array([np.cos(theta), np.sin(theta), dtheta])
                state = np.reshape(state, [1, self.observationDim])
                q_values = self.model.predict(state)
                self.Q[i, j, :] = q_values[0, :]

    # def extractR(self):

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
        # ********* interactive plot ********************* #
        max_Q = np.amax(self.Q, axis=2)
        plt.clf()
        colormap = plt.cm.hot
        # changingData = np.random.rand(10, 12)
        ax = sns.heatmap(max_Q, cmap=colormap)
        rect = patches.Rectangle((self.state[1], self.state[0]), 1, 1, linewidth=1, edgecolor='lime', facecolor='lime')
        ax.add_patch(rect)
        plt.show()
        plt.pause(0.01)
        # ********* interactive plot ********************* #


def main():
    agentCart.learning(([np.pi,0], [np.pi,0]))
    # myDqnAgent.testing()
    # myDqnAgent.extractQ()
    # myDqnAgent.plotQSurface(0, 1, 16)


if __name__ == '__main__':
    agentCart = dqnAgentCart()
    agentCart.render = 1
    main()

