import gym
import numpy as np

import helper

import matplotlib.patches as patches
import seaborn as sns; sns.set()
import random
from collections import deque
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn import linear_model
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import keras
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.kernel_ridge import KernelRidge
'''
Some information about the pendulum environment
Frictionless; Deterministic.

Observation:
obv[0]: cos. range: [-1,1]
obv[1]: sin. range: [-1,1]
obv[2]: dtheta. range: [-8,8]trigo2angle(state_next[0], state_next[1])

Actions: 
range: [-2, 2]

Internal state:
theta: range: [-pi, pi]. Anticlockwise is positive...

Reward:
-(theta^2 + 0.1*dtheta^2 + 0.001*action^2)
=0 when theta = dtheta = action = 0
<0 Otherwise

No discretize. just feed the observation into the network
'''


class DQNSolver:

    def __init__(self, observation_dim, action_number):
        self.exploration_rate = EXPLORATION_MAX
        self.observation_dim = observation_dim
        self.action_number = action_number
        self.memory = deque(maxlen=1000000)
        self.actionSpace =
        self.stateSpace =
        self.action =
        self.state =

    def build(self):
        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(self.observation_dim,), activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(self.action_number, activation="linear"))
        self.model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True))

    def remember(self, state, action, reward, state_next, terminal):
        # self.memory.add((state, action, reward, state_next, terminal))
        self.memory.append((state, action, reward, state_next, terminal))

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update  # action is the index too
            self.model.fit(state, q_values, verbose=0)  # the memory before was kept too!
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def explore(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_number)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])   ## best action index at current state

    def setActionSpace(self, actionSpace):
        self.actionSpace = actionSpace

    def setStateSpace(self, stateSpace):
        self.stateSpace = stateSpace

    # Interface with the environment
    def obvToState(self, obv):  # interface with the environment
        self.state =

    def actionToImpact(self):  # interface with the environment
        impact = self.action
        return impact

    def act(self):
        q_values = self.model.predict(self.state)
        return np.argmax(q_values[0])

    def learn(self):



def pendulum():
    run = 0
    count_valid = 0
    while True:
        run += 1
        state = env.reset(mode = "easy")
        state = np.reshape(state, [1, observation_dim])
        step = 0
        while True:
            step += 1
            # env.render()
            actionID = dqn_solver.explore(state)
            action = action_space[actionID]
            action = np.array([action])
            state_next, reward, terminal, info = env.step(action)
            cumReward.append(cumReward[len(cumReward)-1]+reward)
            theta = helper.trigo2angle(state_next[0], state_next[1])
            # reward = reward + 10 - 10/(1+np.exp(-100*(reward**2-3**2)))
            terminal = 0
            # if theta > np.pi / 3 or theta < -np.pi / 3 or step >= 500:
            if step >= 50:
                terminal = 1
            # reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_dim])
            dqn_solver.remember(state, actionID, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Episode: ", run, " Step: ", step)
                print("Explore rate = ", dqn_solver.exploration_rate)
                break
            dqn_solver.experience_replay()
            env.render()
        if run >= 100:
            count_valid += 1
        else: count_valid = 0
        if count_valid == CONSECUTIVE_WIN:
            break
    env.close()



def generate_Q(theta_space, dtheta_space, action_space, dqn_solver):
    Q = np.zeros((theta_space.size, dtheta_space.size, action_space.size))
    for i,theta in enumerate(theta_space):
        for j, dtheta in enumerate(dtheta_space):
            state = np.array([np.cos(theta), np.sin(theta), dtheta])
            state = np.reshape(state, [1, dqn_solver.observation_dim])
            q_values = dqn_solver.model.predict(state)
            Q[i, j, :] = q_values[0, :]
            # for k, action in enumerate(action_space):
            #     Q[i,j,k] = q_values[0][k]
    return Q


def plot_Q(theta_space, dtheta_space, action_space, Q, fixedThe, fixedDth, fixedAct):
    shapeThe = Q.shape[0]
    shapeDth = Q.shape[1]
    shapeAct = Q.shape[2]
    # fixed action
    mtheta, mdtheta = np.meshgrid(theta_space, dtheta_space)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    q1 = np.reshape(Q[:,:,fixedAct], [shapeThe, shapeDth])
    ax.plot_surface(mtheta.T, mdtheta.T, q1)
    ax.set_xlabel('theta')
    ax.set_ylabel('dtheta')
    ax.set_zlabel('Q value')

    # fixed theta
    mdtheta, maction = np.meshgrid(dtheta_space, action_space)
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    q2 = np.reshape(Q[fixedThe, :, :], [shapeDth, shapeAct])
    ax.plot_surface(mdtheta.T, maction.T, q2)
    ax.set_xlabel('dtheta')
    ax.set_ylabel('action')
    ax.set_zlabel('Q value')
    plt.show()

    # fixed dtheta
    mtheta, maction = np.meshgrid(theta_space, action_space)
    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')
    q3 = np.reshape(Q[:, fixedDth, :], [shapeThe, shapeAct])
    ax.plot_surface(mtheta.T, maction.T, q3)
    ax.set_xlabel('theta')
    ax.set_ylabel('action')
    ax.set_zlabel('Q value')
    plt.show()

    q4 = Q[fixedThe,fixedDth,:]
    plt.plot(action_space, q4)
    plt.show()

    return None


def real_Q(theta_space, dtheta_space, action_space):
    real_Q = np.zeros((theta_space.size, dtheta_space.size, action_space.size))
    post_Q = np.zeros((theta_space.size, dtheta_space.size, action_space.size))
    R = np.zeros((theta_space.size, dtheta_space.size, action_space.size))
    T = np.zeros((theta_space.size, dtheta_space.size, action_space.size, 2))
    # reward table R(theta, dtheta, action)
    # mtheta, mtheta, maction = np.meshgrid(theta_space, dtheta_space, action_space)
    # R = -(mtheta**2 + 0.1*mtheta^2 + 0.001*maction**2)

    # transition table T(theta, dtheta, action)
    # TODO: how to represent T ?

    for i,theta in enumerate(theta_space):
        for j, dtheta in enumerate(dtheta_space):
            for k, action in enumerate(action_space):
                R[i,j,k] = -(theta**2 + 0.1*dtheta**2 + 1*action**2)
                newState = helper.pendulumModel([theta, dtheta], action)
                newStateID = helper.state2index(newState, theta_space, dtheta_space)
                T[i, j, k, :] = newStateID
    ## bellman iteration
    step = 0
    converge = 1000
    plt.ion()  # active interactive mode
    plt.figure(1) # interactive plot of the iteration of Q
    while step < BELLEMAN_STEP and converge > CONVERGE_MARGIN:
        step += 1

        max_Q = np.amax(real_Q, axis=2)
        plt.clf()  # 清空画布上的所有内容
        # changingData = np.random.rand(10, 12)
        ax = sns.heatmap(max_Q)
        plt.show()
        plt.pause(0.01)

        print("bellman step", step)
        print("converge = ", converge)
        for i in range(real_Q.shape[0]):
            for j in range(real_Q.shape[1]):
                for k in range(real_Q.shape[2]):
                    thetaID = int(T[i,j,k,0])
                    dthetaID = int(T[i,j,k,1])
                    # print(thetaID, dthetaID)
                    post_Q[i,j,k] = max_Q[thetaID, dthetaID]
        next_Q = R + GAMMA * post_Q
        converge = np.amax(abs(next_Q - real_Q))
        real_Q = next_Q

    return R, T, real_Q

# in helper
def test(mode):
    env2 = gym.make('Pendulum-v0')
    state = env2.reset(mode=mode)
    state = np.reshape(state, [1, observation_dim])
    print('go into test')
    for step in range(TEST_STEP):
        print("testStep = ", step)
        actionID = dqn_solver.act(state)
        action = action_space[actionID]
        action = np.array([action])
        obv, reward, terminal, info = env2.step(action)
        state_next = np.reshape(obv, [1, observation_dim])
        state = state_next
        env2.render()
    env2.close()
    return None

# in helper
def test2(Q, mode):
    env3 = gym.make('Pendulum-v0')
    obv = env3.reset(mode=mode)
    state = helper.obv2index(obv, theta_space, dtheta_space)
    print('go into test')
    print("stateIndex = ", state)
    for step in range(TEST_STEP):
        print("testStep = ", step)
        actionID = np.argmax(Q[state])
        print("actionIndex = ", actionID)
        action = action_space[actionID]
        action = np.array([action])
        obv, reward, terminal, info = env3.step(action)
        state_next = helper.obv2index(obv, theta_space, dtheta_space)
        state = state_next
        env3.render()
    env3.close()
    return None

if __name__ == '__main__':

    BATCH_SIZE = 200
    GAMMA = 0.8
    EXPLORE_RATE = 0.5
    ENV_NAME = 'Pendulum-v0'
    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.998
    TEST_STEP = 1000
    CONSECUTIVE_WIN = 1
    BELLEMAN_STEP = 1000
    CONVERGE_MARGIN = 1E-10

    action_space = np.linspace(-2, 2, 21, True)
    theta_space = np.linspace(-np.pi, np.pi, 65, False)
    dtheta_space = np.linspace(-8, 8, 65, False)

    cumReward = [0]  # the cumulative reward over all the steps during the training

    env = gym.make(ENV_NAME)
    observation_dim = env.observation_space.shape[0]
    action_number = action_space.shape[0]

    dqn_solver = DQNSolver(observation_dim, action_number)
    dqn_solver.build()
    pendulum()
    # plot_model(dqn_solver.model)
    test("easy")  # mode = easy, 0 or else
    Q = generate_Q(theta_space, dtheta_space, action_space, dqn_solver)
    plot_Q(theta_space, dtheta_space, action_space, Q, 10, 10, 10)

    # R, T, real_Q = real_Q(theta_space, dtheta_space, action_space)
    # plot_Q(theta_space, dtheta_space, action_space, real_Q, 10, 10, 10)
    # test2(real_Q, "easy")