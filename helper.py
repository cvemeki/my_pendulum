# functions for converting the angles
# functions for plotting
import matplotlib.pyplot as plt
import numpy as np
import gym


def trigo2angle(cos, sin):  # from sin and cos to [-pi, pi]
    if sin > 0:  # [0, pi]
        theta = np.arccos(cos)  # arccos returns angle[0, pi]
    else:
        theta = -np.arccos(cos)
    return theta


def state2index(state, theta_space, dtheta_space):
    theta = state[0]
    dtheta = state[1]
    thetaID = discretize(theta, theta_space)
    dthetaID = discretize(dtheta, dtheta_space)
    return np.array([thetaID, dthetaID])  # return the index of the state


def discretize(state, state_space):  # In general discretize a continous item into the discretized space
    for i, dis in enumerate(state_space):  # space from small to big
        if dis > state:
            return i - 1
    return i  # the last interval


def obv2index(obv, theta_space, dtheta_space):
    theta = trigo2angle(obv[0], obv[1])
    theta = discretize(theta, theta_space)
    dtheta = discretize(obv[2], dtheta_space)
    return (theta, dtheta)  # return the index of the state


def visualize_space(space):
    x = np.arange(len(space))
    plt.step(x, np.abs(space), where= 'post')
    plt.grid()
    return None


def pendulumModel(curState, action):
    # a copy from gym environment
    # we can modify the pendulum model here
    th, thdot = curState[0], curState[1]  # th := theta

    g = 10.0
    m = 1.
    l = 1.
    dt = .05

    action = np.clip(action, -2, 2)

    newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * action) * dt
    newth = th + newthdot * dt
    newthdot = np.clip(newthdot, -8, 8)  # pylint: disable=E1111

    cos = np.cos(newth)
    sin = np.sin(newth)

    newth = trigo2angle(cos, sin)
    return np.array([newth, newthdot])


def plotAction(agent):
    bestAction = np.argmax(agent.Q, axis=2)
    colormap = plt.cm.hot
    ax = sns.heatmap(bestAction, cmap=colormap)
    plt.xlabel("dtheta(index)")
    plt.ylabel("theta(index)")
    plt.show()
    return None

def test(mode, env, agent, TEST_STEP):
    obv = env.reset(mode = mode)
    for step in range(TEST_STEP):
        print("testStep = ", step)
        action = agent.act(obv) # do an action based on current state
        obv, _, _, _ = env.step(action)
        env.render()
    env.close()
    return None


'''testing'''
def testing(agent, mode):
    obv = agent.env.reset(mode = mode)
    agent.obvToState(obv)
    for step in range(100):
        agent.act()  # do an action based on current state
        impact = agent.actionToImpact()
        obv, _, _, _ = agent.env.step(impact)
        agent.obvToState(obv)
        agent.interactivePlot()
        agent.env.render()
    agent.env.close()