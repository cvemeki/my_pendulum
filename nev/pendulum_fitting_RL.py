import gym
import numpy as np
import random

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn import linear_model
import math
from keras.models import Sequential
from keras.layers import Dense
import keras
from sklearn.kernel_ridge import KernelRidge
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

We want to find a function f(theta, dtheta, action) = Q, which is continuous...
here we want to minimize the cost...
'''


random.seed = 2019

# class Q_model:
#     def __init__(self, nfeature, power):
#         self.nfeature = nfeature
#         self.power = power  # the power of the features
#         self.weightsDim = 1 + self.power*self.nfeature
#         self.weights = np.zeros((self.weightsDim,1)) # initialize the weights by zeros
#
#     def updateWeights(self, newX, newY):
#         print("newX: ", newX)
#         print("newX shape: ", newX.shape)
#         print("newY: ", newY)
#         print("newY shape: ", newY.shape)
#         print("gradient", self.gradient(newX, newY))
#         self.weights = self.weights - 0.001*self.gradient(newX, newY)
#         print("weights",self.weights)
#
#
#     def gradient(self, x, y):  # x: 1*n; y: 1*1
#         return (x.dot(self.weights)-y)*x.T
#
#     # def SGD(self):
#     #
#     def fit(self, X, Y):  # X: m*n Y: 1*n
#         num_data = X.shape[0]
#         dim_data = X.shape[1]
#         candidates = np.arange(num_data)
#         for i in range(10000):
#             print("fitting: ", i)
#             index = np.random.choice(candidates)
#             newX = X[index:index+1,:]
#             newY = Y[index]
#             self.updateWeights(newX, newY)


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


def computeCost(reward, obv):
    theta = trigo2angle(obv[0], obv[1])
    return -100-10*(reward) + 100/(1+np.exp(-100*(theta**2-(np.pi/3)**2))) ## it's the cost. we want to minimize it


#select an action to explore at current state
def selectAction(current, explore_rate):  # return a real action according to current state
    ## TODO: totally random selection.. Just for collecting the data
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        minID = np.argwhere(Q_table[current] == np.amin(Q_table[current]))
        actionID = random.choice(minID)
        action = continuize(actionID, action_space)
        # action = action_space[actionID]
    return np.array([action])


def continuize (stateID, state_space):
    if stateID+1 < state_space.size:
        return random.choice(np.linspace(state_space[stateID], state_space[stateID+1], 50))
    else:
        return random.choice(np.linspace(state_space[stateID], 2*state_space[stateID] - state_space[stateID-1], 50))


def oneEpisode(learningRate, exploreRate, discountFactor, maxT):
    ##Reset the environment.
    ##Reset the Q-table.
    obv = env.reset(mode="easy")  # begin from the top
    statePre = obv2index(obv)
    thetaPre = trigo2angle(obv[0], obv[1])
    dthetaPre = obv[2]

    for step in range(maxT):
        action = selectAction(statePre, exploreRate)
        actionID = discretize(action, action_space)
        obv, reward, done, _ = env.step(action)
        if trigo2angle(obv[0], obv[1]) > np.pi/3 or trigo2angle(obv[0], obv[1]) < -np.pi/3:
            print(trigo2angle(obv[0], obv[1]))
            break
        cost = computeCost(reward, obv)
        stateNew = obv2index(obv)
        thetaNew = trigo2angle(obv[0], obv[1])
        dthetaNew = obv[2]
        best_q = np.amin(Q_table[stateNew])
        Q_table[statePre + (actionID,)] = (1 - learningRate) * Q_table[statePre + (actionID,)] + learningRate * \
                                          (cost + discountFactor * best_q)
        # y = Q_table[statePre + (actionID,)]
        # x = np.array([[1,thetaPre,dthetaPre,action[0],thetaPre**2,dthetaPre**2,action[0]**2,thetaPre**3,dthetaPre**3,action[0]**3, thetaPre**4,dthetaPre**4,action[0]**4]])
        # Qm.updateWeights(x, y)

        statePre = stateNew
        thetaPre = thetaNew
        dthetaPre = dthetaNew
        # env.render()

    return None

def getXY(Q_table):
    nonZeroID = np.where(Q_table != 0)
    num = nonZeroID[0].shape[0]
    X = np.array([])
    Y = np.array([])
    for i in range(0,num):
        x = np.array([])
        theta = theta_space[nonZeroID[0][i]]
        dtheta = dtheta_space[nonZeroID[1][i]]
        action = action_space[nonZeroID[2][i]]
        AB = theta*dtheta
        AC = theta*action
        BC = dtheta*action
        ABC = theta*dtheta*action
        AAB = theta**2*dtheta
        AAC = theta**2*action
        ABB = theta*dtheta**2
        BBC = dtheta**2*action
        ACC = theta*action**2
        BCC = dtheta*action**2

        x = np.append(x, 1)
        for p in range(3):
            if p == 0: # power1
                x = np.append(x, theta)
                x = np.append(x, dtheta)
                x = np.append(x, action)

            if p == 1: # power2
                x = np.append(x, theta**2)
                x = np.append(x, dtheta**2)
                x = np.append(x, action**2)
                x = np.append(x, AB)
                x = np.append(x, AC)
                x = np.append(x, BC)
                # power 1/3
                x = np.append(x, abs(theta) / theta * abs(theta) ** (1 / 3))
                x = np.append(x, abs(dtheta) / dtheta * abs(dtheta) ** (1 / 3))
                x = np.append(x, abs(action) / action * abs(action) ** (1 / 3))

            if p == 2: #power3
                x = np.append(x, theta ** 3)
                x = np.append(x, dtheta ** 3)
                x = np.append(x, action ** 3)
                x = np.append(x, ABC)
                x = np.append(x, AAB)
                x = np.append(x, AAC)
                x = np.append(x, ABB)
                x = np.append(x, BBC)
                x = np.append(x, ACC)
                x = np.append(x, BCC)

            # if p%2 != 0: # p = 1, 3, 5...
            #     x = np.append(x, abs(theta)/theta * abs(theta) ** (1/(p + 2)))
            #     x = np.append(x, abs(dtheta)/dtheta * abs(dtheta) ** (1/(p + 2)))
            #     x = np.append(x, abs(action)/action * abs(action) ** (1/(p + 2)))
            #     x = np.append(x, abs(AB) / AB * abs(AB) ** (1 / (p + 2)))
            #     x = np.append(x, abs(AC) / AC * abs(AC) ** (1 / (p + 2)))
            #     x = np.append(x, abs(BC) / BC * abs(BC) ** (1 / (p + 2)))
            #     x = np.append(x, abs(ABC) / ABC * abs(ABC) ** (1 / (p + 2)))

        x = np.expand_dims(x, axis=1)
        x = x.T
        y = np.array([Q_table[nonZeroID[0][i],nonZeroID[1][i],nonZeroID[2][i]]])
        y = np.expand_dims(y, axis=1)
        if i == 0:
            X = x
            Y = y
        else:
            X = np.append(X, x, axis = 0)
            Y = np.append(Y, y, axis = 0)
        print("shape X: ", X.shape)
        print("shape x: ", x.shape)
        print("shape Y: ", Y.shape)
        print("shape y: ", y.shape)
    return X,Y


def getX_test():
    X = np.array([])
    count = 0
    for theta in theta_space:
        for dtheta in dtheta_space:
            for action in action_space:
                AB = theta * dtheta
                AC = theta * action
                BC = dtheta * action
                ABC = theta * dtheta * action
                AAB = theta ** 2 * dtheta
                AAC = theta ** 2 * action
                ABB = theta * dtheta ** 2
                BBC = dtheta ** 2 * action
                ACC = theta * action ** 2
                BCC = dtheta * action ** 2

                x = np.array([])
                x = np.append(x, 1)
                for p in range(3):
                    if p == 0:  # power1
                        x = np.append(x, theta)
                        x = np.append(x, dtheta)
                        x = np.append(x, action)

                    if p == 1:  # power2
                        x = np.append(x, theta ** 2)
                        x = np.append(x, dtheta ** 2)
                        x = np.append(x, action ** 2)
                        x = np.append(x, AB)
                        x = np.append(x, AC)
                        x = np.append(x, BC)
                        # power 1/3
                        x = np.append(x, abs(theta) / theta * abs(theta) ** (1 / 3))
                        x = np.append(x, abs(dtheta) / dtheta * abs(dtheta) ** (1 / 3))
                        x = np.append(x, abs(action) / action * abs(action) ** (1 / 3))

                    if p == 2:  # power3
                        x = np.append(x, theta ** 3)
                        x = np.append(x, dtheta ** 3)
                        x = np.append(x, action ** 3)
                        x = np.append(x, ABC)
                        x = np.append(x, AAB)
                        x = np.append(x, AAC)
                        x = np.append(x, ABB)
                        x = np.append(x, BBC)
                        x = np.append(x, ACC)
                        x = np.append(x, BCC)

                    # x = np.append(x, theta ** (p + 1))
                    # x = np.append(x, dtheta ** (p + 1))
                    # x = np.append(x, action ** (p + 1))
                    # x = np.append(x, AB ** (p + 1))
                    # x = np.append(x, AC ** (p + 1))
                    # x = np.append(x, BC ** (p + 1))
                    # x = np.append(x, ABC ** (p + 1))
                    #
                    # if p % 2 != 0:  # p = 1, 3, 5...
                    #     x = np.append(x, abs(theta) / theta * abs(theta) ** (1 / (p + 2)))
                    #     x = np.append(x, abs(dtheta) / dtheta * abs(dtheta) ** (1 / (p + 2)))
                    #     x = np.append(x, abs(action) / action * abs(action) ** (1 / (p + 2)))
                    #     x = np.append(x, abs(AB) / AB * abs(AB) ** (1 / (p + 2)))
                    #     x = np.append(x, abs(AC) / AC * abs(AC) ** (1 / (p + 2)))
                    #     x = np.append(x, abs(BC) / BC * abs(BC) ** (1 / (p + 2)))
                    #     x = np.append(x, abs(ABC) / ABC * abs(ABC) ** (1 / (p + 2)))

                x = np.expand_dims(x, axis=1)
                x = x.T
                if count == 0:
                    X = x
                else:
                    X = np.append(X, x, axis=0)
                count += 1
                print("count_getX_test = ", count)
    return X


def getQ_test(Y):
    Q = 0*np.copy(Q_table)
    count = 0
    for i in range(theta_space.shape[0]):
        for j in range(dtheta_space.shape[0]):
            for k in range(action_space.shape[0]):
                Q[i,j,k] = Y[count]
                count += 1
    return Q


def simulate():
    # define some simulation parameters here

    learningRate = LEARNING_RATE
    exploreRate = EXPLORE_RATE
    discountFactor = DISCOUNT_FACTOR
    maxT = MAX_T
    numEpisode = NUM_EPISODES

    for episode in range(numEpisode):
        print('episode:%s', episode)
        oneEpisode(learningRate, exploreRate, discountFactor, maxT)
    return None


def test_pendulum(Q, mode = "0"):
    env2 = gym.make('Pendulum-v0')
    obv = env2.reset(mode=mode)
    statePre = obv2index(obv)
    print('go into test')
    for step in range(TEST_STEP):
        print('testStep=', step)
        actionID = np.argmin(Q[statePre])
        # action = action_space[actionID]
        action = continuize(actionID, action_space)
        action = [action]
        print('action', action)
        obv, reward, done, _ = env2.step(action)
        stateNew = obv2index(obv)
        print('theta', theta_space[stateNew[0]])
        statePre = stateNew
        env2.render()
    env2.close()
    return None

# def test(dthetaID):


if __name__ == '__main__':

    '''State-action space'''
    # portion_theta = np.pi / 24
    # portion_dtheta = 0.5
    portion_action = 0.1

    num_theta = 33
    num_dtheta = 33
    theta_min = -np.pi/2
    theta_max = np.pi/2

    # theta_space = np.arange(theta_min, theta_max, portion_theta)
    # dtheta_space = np.arange(-8.0, 8.0, portion_dtheta)
    action_space = np.arange(-2.0, 2.0, portion_action)
    theta_space = np.linspace(theta_min, theta_max, num_theta, False)
    dtheta_space = np.linspace(-8, 8, num_dtheta, False)

    '''Initialization'''
    Q_table = -10000*np.zeros((theta_space.size, dtheta_space.size, action_space.size))
    T_table = -10000*np.ones((theta_space.size, dtheta_space.size, action_space.size, 2))
    R_table = np.zeros((theta_space.size, dtheta_space.size, action_space.size))
    # Qm = Q_model(3, 4)

    '''Some simulation hyper-parameters'''
    LEARNING_RATE = 0.8  # in deterministic case, we just need to use 1
    EXPLORE_RATE = 0.5
    DISCOUNT_FACTOR = 0.9
    MAX_T = 1000
    NUM_EPISODES = 1000

    '''Begin simulation'''
    env = gym.make('Pendulum-v0')
    simulate()
    env.close()

    '''Generate Q_table offline'''
    # Q_table = -10000 * np.zeros((theta_space.size, dtheta_space.size, action_space.size))
    # Q_table_final = generateQ()

    '''get XY and fit model'''
    X_train,Y_train = getXY(Q_table)
    # Qm.fit(X_train, Y_train)

    reg = linear_model.Ridge(alpha=.2)
    reg.fit(X_train, Y_train[:,0])

    # reg = linear_model.SGDRegressor(max_iter=10000, tol=1e-3)
    # reg.fit(X_train, Y_train[:,0])

    # model = Sequential()
    # model.add(Dense(units=64, activation='selu', input_dim=X_train.shape[1]))
    # model.add(Dense(units=64, activation='selu'))
    # model.add(Dense(units=1, activation='softmax'))
    # model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True), metrics=['mae'])
    # model.fit(X_train, Y_train, epochs=1000, batch_size=32)
    # loss_and_metrics = model.evaluate(X_train, Y_train, batch_size=128)

    # clf = KernelRidge(alpha=1.0)
    # clf.fit(X_train, Y_train[:,0])

    '''Test model'''
    X_test = getX_test()
    Y_test = reg.predict(X_test)

    '''Test with pendulum'''
    TEST_STEP = 500
    Q_test = getQ_test(Y_test) ## according to Y_test
    test_pendulum(Q_test)


    '''plot Q table'''
    # Y_pre = reg.predict(X_train)
    # Y_pre = model.predict(X_train, batch_size=128)
    Y_pre = reg.predict(X_train)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    theta_train = X_train[:,1]
    dtheta_train = X_train[:,2]
    action_train = X_train[:,3]
    theta_train = theta_train[np.where(dtheta_train == dtheta_space[16])]
    action_train = action_train[np.where(dtheta_train == dtheta_space[16])]
    Y_train = Y_train[np.where(dtheta_train == dtheta_space[16])]
    Y_pre = Y_pre[np.where(dtheta_train == dtheta_space[16])]

    theta_test = X_test[:, 1]
    dtheta_test = X_test[:, 2]
    action_test = X_test[:, 3]
    theta_test = theta_test[np.where(dtheta_test == dtheta_space[16])]
    action_test = action_test[np.where(dtheta_test == dtheta_space[16])]
    y_test = Y_test[np.where(dtheta_test == dtheta_space[16])]

    ax.scatter(theta_train, action_train, Y_train)
    ax.scatter(theta_train, action_train, Y_pre)
    ax.scatter(theta_test, action_test, y_test)
    plt.show()
