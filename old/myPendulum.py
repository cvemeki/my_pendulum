import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns; sns.set()

'''
Some information about the pendulum environment
Frictionless; Deterministic.

Observation:
obv[0]: cos. range: [-1,1]
obv[1]: sin. range: [-1,1]
obv[2]: dtheta. range: [-8,8]

Actions: 
range: [-2, 2]

Internal state:
theta: range: [-pi, pi]. Anticlockwise is positive...

Reward:
-(theta^2 + 0.1*dtheta^2 + 0.001*action^2)
=0 when theta = dtheta = action = 0
<0 Otherwise
'''
random.seed = 1234
test_global = 5566


# class qAgentPend:
#
#     def __init__(self):
#         self.memory = deque(maxlen=400) # default = 1000000
#         '''changeable parameters'''
#
#         '''environment'''
#         self.env = gym.make('Pendulum-v0')
#         # self.actionSpace = np.linspace(-2, 2, 21, True)
#         self.actionSpace = np.array([-1,1,0])  # what if binary action(bang-bang control)?
#         self.stateSpace = (np.linspace(-np.pi, np.pi, 65, False), np.linspace(-8, 8, 65, False)) # it's a tuple
#         self.stateSpace1 = self.stateSpace[0]
#         self.stateSpace2 = self.stateSpace[1]
#
#         '''constants'''
#
#         '''neural network model'''
#
#         '''learning'''
#         self.BATCH_SIZE = 200
#         self.GAMMA = 0.9
#         self.STEPS_PER_EPISODE_MAX = 200
#         self.STEPS_PER_EPISODE_WHEN_SUCCESS = 50
#
#         '''exploring'''
#         self.EXPLORE_RATE = 0.5
#         self.EXPLORE_MAX = 1.0
#         self.EXPLORE_MIN = 0.01
#         self.EXPLORE_DECAY = 0.999
#         self.LEARNING_RATE = 0.99
#
#         '''testing'''
#         self.TEST_STEP = 1000
#
#         '''just initialization'''
#
#         self.render = 0
#         '''learning parameters'''
#         self.nLearningStep = 0
#         self.nLearningEpisode = 0
#         self.nSuccessfulEpisode = 0
#         self.nTestingStep = 0
#         self.terminateEpisode = 0
#         self.terminateLearning = 0
#
#         '''internal'''
#         self.action = self.actionSpace[0]
#         self.state = (1, 1)  # a tuple
#         self.oldState = (1, 1)  # a tuple
#
#         '''model'''
#         self.observationDim = self.env.observation_space.shape[0]  # = 3
#         self.actionNumber = self.actionSpace.shape[0]
#         self.Q = np.zeros((self.stateSpace1.size, self.stateSpace2.size, self.actionSpace.size))
#         print("Milestone: initialized finished")
#
#     '''Internal mechanisms'''
#     def remember(self, state, action, reward, state_next, terminal):
#         self.memory.append((state, action, reward, state_next, terminal))
#
#     def experienceReplay(self):
#         if len(self.memory) < self.BATCH_SIZE:
#             return
#         batch = random.sample(self.memory, self.BATCH_SIZE)
#         for state, action, reward, state_next, terminal in batch:
#             q_update = reward
#             if not terminal:
#                 q_update = (reward + self.GAMMA * np.amax(self.Q[state_next, :]))  #np.amax(self.model.predict(state_next)[0]))
#             self.Q[state + (action,)] = (1 - self.LEARNING_RATE) * self.Q[state + (action,)] + self.LEARNING_RATE * q_update
#
#     '''make actions'''
#     def act(self):
#         self.action = np.argmax(self.Q[self.state])
#
#     def explore(self):
#         if np.random.rand() < self.EXPLORE_RATE:
#             self.action = random.randrange(self.actionNumber)
#             return
#         maxID = np.argwhere(self.Q[self.state] == np.amax(self.Q[self.state]))
#         self.action = random.choice(maxID)
#
#     '''Interact with environment'''
#     def obvToState(self, obv):  # interface with the environment
#         self.state = helper.obv2index(obv, self.stateSpace1, self.stateSpace2)
#
#     def actionToImpact(self):  # interface with the environment
#         impact = (self.actionSpace[self.action], )
#         return impact
#
#     '''learning'''
#     def oneLearningStep(self):
#         self.nLearningStep += 1
#         print("learning: Episode = ", self.nLearningEpisode, "; Step = ", self.nLearningStep, "; explore rate = ", self.EXPLORE_RATE)
#         self.explore()
#         impact = self.actionToImpact()
#         self.oldState = self.state
#         obv, reward, _, info = self.env.step(impact)
#         print(obv)
#         self.obvToState(obv)  # state = state_next
#         # if self.nLearningStep >= self.STEPS_PER_EPISODE_MAX:
#         #     self.terminateEpisode = 1
#         # reward and terminate condition
#         if np.linalg.norm([helper.trigo2angle(obv[0], obv[1]), obv[2]]) < 0.1 or self.nLearningStep > 200:
#             self.terminateEpisode = 1
#             # reward = reward + 100
#
#         # q_update = (reward + self.GAMMA * np.amax(self.Q[self.state, :]))  # np.amax(self.model.predict(state_next)[0]))
#         # self.Q[self.oldState + (self.action,)] = (1 - self.LEARNING_RATE) * self.Q[self.state + (self.action,)] + self.LEARNING_RATE * q_update
#
#         self.remember(self.oldState, self.action, reward, self.state, self.terminateEpisode)
#         self.experienceReplay()
#
#         if self.render:
#             self.env.render()
#         self.EXPLORE_RATE *= self.EXPLORE_DECAY
#         self.EXPLORE_RATE = max(self.EXPLORE_MIN, self.EXPLORE_RATE)
#
#     def oneLearningEpisode(self, mode):
#         self.nLearningEpisode += 1
#         obv = self.env.reset(mode=mode)
#         self.obvToState(obv)
#         self.nLearningStep = 0
#         self.terminateEpisode = 0
#         while not self.terminateEpisode:
#             self.oneLearningStep()
#         print("one episode ends")
#         # if self.nLearningStep >= self.STEPS_PER_EPISODE_WHEN_SUCCESS:
#         #     self.nSuccessfulEpisode += 1
#
#     def learning(self, mode):
#         self.nLearningEpisode = 0
#         # while self.nSuccessfulEpisode < 100:
#         while self.nLearningEpisode < 200:
#             self.EXPLORE_RATE = 0.5  # reset the learning rate for every episode. (otherwise it's too low for the following episode)
#             self.oneLearningEpisode(mode)
#         self.env.close()
#
#     '''testing'''
#     def testing(self, mode):
#         obv = self.env.reset(mode = mode)
#         self.obvToState(obv)
#         for step in range(self.TEST_STEP):
#             self.act()  # do an action based on current state
#             impact = self.actionToImpact()
#             obv, _, _, _ = self.env.step(impact)
#             self.obvToState(obv)
#             self.interactivePlot()
#             if self.render:
#                 self.env.render()
#         self.env.close()
#
#
#     '''plotting'''
#     def plotQSurface(self, xDim, yDim, anchor):
#         if {xDim, yDim} == {0,1}:
#             slicedQ = np.reshape(self.Q[:, :, anchor], [self.Q.shape[0], self.Q.shape[1]])
#             xAxis, yAxis = np.meshgrid(self.stateSpace1, self.stateSpace2)
#             xLabel, yLabel = 'theta', 'dtheta'
#
#         elif {xDim, yDim} == {0,2}:
#             slicedQ = np.reshape(self.Q[:, anchor, :], [self.Q.shape[0], self.Q.shape[2]])
#             xAxis, yAxis = np.meshgrid(self.stateSpace1, self.actionSpace)
#             xLabel, yLabel = 'theta', 'action'
#
#         else: # {xDim, yDim} == {1,2}
#             slicedQ = np.reshape(self.Q[anchor, :, :], [self.Q.shape[1], self.Q.shape[2]])
#             xAxis, yAxis = np.meshgrid(self.stateSpace2, self.actionSpace)
#             xLabel, yLabel = 'dtheta', 'action'
#
#         fig1 = plt.figure()
#         ax = fig1.gca(projection='3d')
#         ax.plot_surface(xAxis.T, yAxis.T, slicedQ)
#         ax.set_xlabel(xLabel)
#         ax.set_ylabel(yLabel)
#         ax.set_zlabel('Q value')
#         plt.show()
#
#     def interactivePlot(self):
#         # ********* interactive plot ********************* #
#         max_Q = np.amax(self.Q, axis=2)
#         plt.clf()
#         colormap = plt.cm.hot
#         # changingData = np.random.rand(10, 12)
#         ax = sns.heatmap(max_Q, cmap=colormap)
#         rect = patches.Rectangle((self.state[1], self.state[0]), 1, 1, linewidth=1, edgecolor='lime', facecolor='lime')
#         ax.add_patch(rect)
#         plt.show()
#         plt.pause(0.01)
#         # ********* interactive plot ********************* #
#
#
# class qAgentCart:
#     def __init__(self):
#         self.memory = deque(maxlen=1000000)
#         '''changeable parameters'''
#
#         '''environment'''
#         self.env = gym.make("CartPole-v1")
#         # self.actionSpace = np.linspace(-2, 2, 21, True)
#         # self.actionSpace = np.array([-1,1,0])  # what if binary action(bang-bang control)?
#         self.actionSpace = np.array([0,1])
#         self.stateSpace = (np.linspace(-np.pi, np.pi, 65, False), np.linspace(-8, 8, 65, False)) # it's a tuple
#         self.stateSpace1 = self.stateSpace[0]
#         self.stateSpace2 = self.stateSpace[1]
#
#         '''constants'''
#
#         '''neural network model'''
#
#         '''learning'''
#         self.BATCH_SIZE = 20
#         self.GAMMA = 0.95
#         self.STEPS_PER_EPISODE_MAX = 200
#         self.STEPS_PER_EPISODE_WHEN_SUCCESS = 50
#
#         '''exploring'''
#         self.EXPLORE_RATE_INIT = 0.5
#         self.EXPLORE_RATE = self.EXPLORE_RATE_INIT
#         self.EXPLORE_MAX = 1.0
#         self.EXPLORE_MIN = 0.01
#         self.EXPLORE_DECAY = 0.995
#         self.LEARNING_RATE = 0.99
#
#         '''testing'''
#         self.TEST_STEP = 1000
#
#         '''just initialization'''
#
#         self.render = 1
#         '''learning parameters'''
#         self.nLearningStep = 0
#         self.nLearningEpisode = 0
#         self.nSuccessfulEpisode = 0
#         self.nTestingStep = 0
#         self.terminateEpisode = 0
#         self.terminateLearning = 0
#
#         '''internal'''
#         self.action = self.actionSpace[0]
#         self.state = (self.stateSpace1[0], self.stateSpace2[0])  # a tuple
#
#         '''model'''
#         self.observationDim = self.env.observation_space.shape[0]  # = 3
#         self.actionNumber = self.actionSpace.shape[0]
#         self.Q = np.zeros((self.stateSpace1.size, self.stateSpace2.size, self.actionSpace.size))
#         print("Milestone: initialized finished")
#
#     '''Internal mechanisms'''
#     def remember(self, state, action, reward, state_next, terminal):
#         self.memory.append((state, action, reward, state_next, terminal))
#
#     def experienceReplay(self):
#         if len(self.memory) < self.BATCH_SIZE:
#             return
#         batch = random.sample(self.memory, self.BATCH_SIZE)
#         for state, action, reward, state_next, terminal in batch:
#             q_update = reward
#             if not terminal:
#                 q_update = (reward + self.GAMMA * np.amax(self.Q[state_next, :]))  # np.amax(self.model.predict(state_next)[0]))
#             self.Q[state + (action,)] = (1 - self.LEARNING_RATE) * self.Q[state + (action,)] + self.LEARNING_RATE * q_update
#
#     '''make actions'''
#     def act(self):
#         self.action = np.argmax(self.Q[self.state])
#
#     def explore(self):
#         if np.random.rand() < self.EXPLORE_RATE:
#             self.action = random.randrange(self.actionNumber)
#             return
#         q_values = self.model.predict(self.state)
#         self.action = np.argmax(q_values[0])  # best action index at current state
#
#     '''Interact with environment'''
#     def obvToState(self, obv):  # interface with the environment
#         self.state = np.reshape(obv, [1, self.observationDim])
#
#     def actionToImpact(self):  # interface with the environment
#         # impact = (self.actionSpace[self.action], )
#         impact = self.actionSpace[self.action]
#         return impact
#
#     '''learning'''
#     def oneLearningStep(self):
#         self.nLearningStep += 1
#         # print("learning: Episode = ", self.nLearningEpisode, "; Step = ", self.nLearningStep, "; explore rate = ", self.EXPLORE_RATE)
#         # print(self.state)
#         self.explore()
#         impact = self.actionToImpact()
#         oldState = self.state
#         obv, reward, self.terminateEpisode, info = self.env.step(impact)
#         self.obvToState(obv)  # state = state_next
#         reward = reward if not self.terminateEpisode else -reward
#         # if self.nLearningStep >= self.STEPS_PER_EPISODE_MAX:
#         #     self.terminateEpisode = 1
#         # reward and terminate condition
#         # if np.linalg.norm([helper.trigo2angle(obv[0], obv[1]), obv[2]]) < 0.1:
#         #     self.terminateEpisode = 1
#         #     reward = reward + 100
#         self.remember(oldState, self.action, reward, self.state, self.terminateEpisode)
#         self.experienceReplay()
#         if self.render:
#             self.env.render()
#         self.EXPLORE_RATE *= self.EXPLORE_DECAY
#         self.EXPLORE_RATE = max(self.EXPLORE_MIN, self.EXPLORE_RATE)
#
#     def oneLearningEpisode(self, mode):
#         self.nLearningEpisode += 1
#         # obv = self.env.reset(mode=mode)
#         obv = self.env.reset()
#         self.obvToState(obv)
#         self.nLearningStep = 0
#         self.terminateEpisode = 0
#         while not self.terminateEpisode:
#             self.oneLearningStep()
#         # print("one episode ends")
#         # if self.nLearningStep >= self.STEPS_PER_EPISODE_WHEN_SUCCESS:
#         #     self.nSuccessfulEpisode += 1
#
#     def learning(self, mode):
#         self.nLearningEpisode = 0
#         # while self.nSuccessfulEpisode < 100:
#         while self.nLearningEpisode < 100:
#             print("episode = ", self.nLearningEpisode, " Step = ", self.nLearningStep)
#             # self.EXPLORE_RATE = 0.5  # reset the learning rate for every episode. (otherwise it's too low for the following episode)
#             self.oneLearningEpisode(mode)
#         self.env.close()
#
#     '''testing'''
#     def testing(self, mode):
#         obv = self.env.reset(mode = mode)
#         # obv = self.env.reset()
#         for step in range(self.TEST_STEP):
#             self.obvToState(obv)
#             self.act()  # do an action based on current state
#             impact = self.actionToImpact()
#             obv, _, _, _ = self.env.step(impact)
#             if self.render:
#                 self.env.render()
#         self.env.close()
#
#     '''extract the Q matrix from the model'''
#     def extractQ(self):
#         for i, theta in enumerate(self.stateSpace1):
#             for j, dtheta in enumerate(self.stateSpace2):
#                 state = np.array([np.cos(theta), np.sin(theta), dtheta])
#                 state = np.reshape(state, [1, self.observationDim])
#                 q_values = self.model.predict(state)
#                 self.Q[i, j, :] = q_values[0, :]
#
#     # def extractR(self):
#
#     '''plotting'''
#     def plotQSurface(self, xDim, yDim, anchor):
#         if {xDim, yDim} == {0,1}:
#             slicedQ = np.reshape(self.Q[:, :, anchor], [self.Q.shape[0], self.Q.shape[1]])
#             xAxis, yAxis = np.meshgrid(self.stateSpace1, self.stateSpace2)
#             xLabel, yLabel = 'theta', 'dtheta'
#
#         elif {xDim, yDim} == {0,2}:
#             slicedQ = np.reshape(self.Q[:, anchor, :], [self.Q.shape[0], self.Q.shape[2]])
#             xAxis, yAxis = np.meshgrid(self.stateSpace1, self.actionSpace)
#             xLabel, yLabel = 'theta', 'action'
#
#         else: # {xDim, yDim} == {1,2}
#             slicedQ = np.reshape(self.Q[anchor, :, :], [self.Q.shape[1], self.Q.shape[2]])
#             xAxis, yAxis = np.meshgrid(self.stateSpace2, self.actionSpace)
#             xLabel, yLabel = 'dtheta', 'action'
#
#         fig1 = plt.figure()
#         ax = fig1.gca(projection='3d')
#         ax.plot_surface(xAxis.T, yAxis.T, slicedQ)
#         ax.set_xlabel(xLabel)
#         ax.set_ylabel(yLabel)
#         ax.set_zlabel('Q value')
#         plt.show()
#
#     def interactivePlot(self):
#         max_Q = np.amax(self.Q, axis=2)
#         plt.clf()
#         colormap = plt.cm.hot
#         # changingData = np.random.rand(10, 12)
#         ax = sns.heatmap(max_Q, cmap=colormap)
#         rect = patches.Rectangle((self.state[1], self.state[0]), 1, 1, linewidth=1, edgecolor='lime', facecolor='lime')
#         ax.add_patch(rect)
#         plt.show()
#         plt.pause(0.01)




##transfer a continue observation to a discretize one.
def trigo2angle(cos, sin):  # from sin and cos to [-pi, pi]
    if sin > 0:  # [0, pi]
        theta = np.arccos(cos)  # arccos returns angle[0, pi]
    else:
        theta = -np.arccos(cos)
    return theta


def discretize(state, state_space):  # In general discretize a continous item into the discretized space
    for i, dis in enumerate(state_space):  # space from small to big
        if dis > state:
            return i-1
    return i  # the last interval


def obv2index(obv):
    theta = trigo2angle(obv[0], obv[1])
    theta = discretize(theta, theta_space)
    dtheta = discretize(obv[2], dtheta_space)
    return tuple([theta, dtheta])  # return the index of the state


def computeReward(reward, obv):
    theta = trigo2angle(obv[0], obv[1])
    return 100 - 10*reward - 100/(1+np.exp(-100*(theta**2-(np.pi/3)**2)))   #100/​(1+​np.exp(​(‑100)*​(x^​2-​1)))


#select an action to explore at current state
def selectAction(current, explore_rate):  # return a real action according to current state
    if random.random() < explore_rate:
        action = env.action_space.sample()
    #     candidateAllowed = np.argwhere(Q_table[current] >= 0)
    #     if candidateAllowed.size > 0:
    #         actionID = random.choice(candidateAllowed)
    #         action = continuize(actionID, action_space)
    #         # action = action_space[actionID]
    #     else:
    #         action = env.action_space.sample()

    else:
        maxID = np.argwhere(Q_table[current] == np.amax(Q_table[current]))
        actionID = random.choice(maxID)
        action = continuize(actionID, action_space)
        # action = action_space[actionID]
    return np.array([action])


def continuize (stateID, state_space):
    if stateID+1 < state_space.size:
        return random.choice(np.linspace(state_space[stateID], state_space[stateID+1], 50))
    else:
        return random.choice(np.linspace(state_space[stateID], 2*state_space[stateID] - state_space[stateID-1], 50))

def simulate():
    # define some simulation parameters here
    episode = 0
    learningRate = LEARNING_RATE
    exploreRate = EXPLORATION_MAX
    discountFactor = DISCOUNT_FACTOR
    # maxT = MAX_T
    # numEpisode = NUM_EPISODES
    count_valid = 0
    plt.ion()  # active interactive mode
    plt.figure(1)  # interactive plot of the iteration of Q
    while True:
        print('episode:%s', episode)
        episode += 1
        obv = env.reset(mode = ([0,0], [0,0]))  # begin from the top
        statePre = obv2index(obv)
        step = 0
        while True:
            step += 1

            action = selectAction(statePre, exploreRate)
            actionID = discretize(action, action_space)
            obv, reward, terminal, _ = env.step(action)

            theta = trigo2angle(obv[0], obv[1])
            terminal = 0
            if theta > np.pi / 3 or theta < -np.pi / 3 or step >= 500:
                terminal = 1
            # reward = reward + 10
            cumReward.append(cumReward[len(cumReward) - 1] + reward)
            if terminal == 1:
                print(theta)
                print("Episode: ", episode, " Step: ", step)
                print("explore = ", exploreRate)
                break
            stateNew = obv2index(obv)
            best_q = np.amax(Q_table[stateNew])
            Q_table[statePre + (actionID,)] = (1 - learningRate) * Q_table[statePre + (actionID,)] + learningRate * \
                                              (reward + discountFactor * best_q)

            # ********* interactive plot ********************* #
            max_Q = np.amax(Q_table, axis=2)
            plt.clf()
            colormap = plt.cm.hot
            # changingData = np.random.rand(10, 12)
            ax = sns.heatmap(max_Q, cmap=colormap)
            rect = patches.Rectangle((statePre[1], statePre[0]), 1, 1, linewidth=1, edgecolor='lime', facecolor='lime')
            ax.add_patch(rect)
            plt.show()
            plt.pause(0.01)
            # ********* interactive plot ********************* #

            ##Set state to current state
            statePre = stateNew
            exploreRate *= EXPLORATION_DECAY
            exploreRate = max(EXPLORATION_MIN, exploreRate)
            env.render()
        if step >= 200:
            count_valid += 1
        else: count_valid = 0
        if count_valid == 2:
            break
    return None

#
# def generateQ():  # offline generate the matrix Q by the simulation result
#     minIter = 100
#     n = 0
#     converge = 100
#     while n < minIter and converge > 1E-5:
#         q_table0 = np.copy(Q_table)
#         for i in range(Q_table.shape[0]):
#             for j in range(Q_table.shape[1]):
#                 for k in range(Q_table.shape[2]):
#                     best_q = np.amax(Q_table, axis=2)
#                     stateID1 = int(T_table[i, j, k, 0])
#                     stateID2 = int(T_table[i, j, k, 1])
#                     if stateID1 != -10000:  # combination i,j,k have been visited
#                         Q_table[i, j, k] = R_table[i, j, k] + 0.8*best_q[stateID1, stateID2]
#         converge = np.max(np.abs(Q_table-q_table0))
#         n = n+1
#
#     return Q_table


def test():
    env2 = gym.make('Pendulum-v0')
    obv = env2.reset(mode = ([0,0], [0,0]))
    statePre = obv2index(obv)
    print('go into test')
    for step in range(TEST_STEP):
        print('testStep=', step)
        actionID = np.argmax(Q_table[statePre])
        # action = action_space[actionID]
        action = continuize(actionID, action_space)
        action = [action]
        print('action', action)

        obv, reward, done, _ = env2.step(action)
        stateNew = obv2index(obv)
        theta = trigo2angle(obv[0], obv[1])
        if theta > theta_max or theta < theta_min:
            break
        print('theta', theta)
        statePre = stateNew
        env2.render()
    env2.close()
    return None


if __name__ == '__main__':

    '''State-action space'''
    # portion_theta = np.pi / 24
    # portion_dtheta = 0.5
    portion_action = 0.1

    num_theta = 33
    num_dtheta = 16
    theta_min = -np.pi
    theta_max = np.pi

    # theta_space = np.arange(theta_min, theta_max, portion_theta)
    # dtheta_space = np.arange(-8.0, 8.0, portion_dtheta)
    action_space = np.arange(-2.0, 2.0, portion_action)
    # action_space = np.linspace(-2, 2, 41, False)
    theta_space = np.linspace(theta_min, theta_max, num_theta, False)
    dtheta_space = np.linspace(-8, 8, num_dtheta, False)

    '''Initialization'''
    Q_table = -16*np.ones((theta_space.size, dtheta_space.size, action_space.size))
    # T_table = -10000*np.ones((theta_space.size, dtheta_space.size, action_space.size, 2))
    # R_table = np.zeros((theta_space.size, dtheta_space.size, action_space.size))

    '''Some simulation hyper-parameters'''
    LEARNING_RATE = 0.8  # in deterministic case, we just need to use 1
    # EXPLORE_RATE = 0.6
    DISCOUNT_FACTOR = 0.9
    MAX_T = 2000
    NUM_EPISODES = 2000

    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.99995
    exploreRate = EXPLORATION_MAX

    cumReward = [0]

    '''Begin simulation'''
    env = gym.make('Pendulum-v0')
    simulate()
    env.close()

    '''Generate Q_table offline'''
    # Q_table = -10000 * np.zeros((theta_space.size, dtheta_space.size, action_space.size))
    # Q_table_final = generateQ()

    '''Test'''
    TEST_STEP = 1000
    test()
