import gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from math import *
np.random.seed(0)
import seaborn as sns; sns.set()

def test():
    env = gym.make('Pendulum-v0')
    obv = env.reset(mode = "0")
    print('go into test')
    for step in range(1):
        print("step=", step)
        print("cos=", obv[0])
        print("sin=", obv[1])
        print("dth", obv[2])
        action = [-0]
        print("step=", step)
        print("cos=", obv[0])
        print("sin=", obv[1])
        print("dth", obv[2])
        obv, reward, terminal, info = env.step(action)
        env.render()
    env.close()
    return None


def trigo2angle(cos, sin):  # from sin and cos to [-pi, pi]
    if sin > 0:  # [0, pi]
        theta = np.arccos(cos)  # arccos returns angle[0, pi]
    else:
        theta = -np.arccos(cos)
    return theta

class testPointer:
    def __init__(self):
        self.state = 1
        self.oldState = 2

    def equalState(self):
        self.oldState = self.state
        self.state = 3

'''testing'''
def testing(agent, mode, step):
    obv = agent.env.reset(mode = mode)
    agent.obvToState(obv)
    for step in range(step):
        agent.act()  # do an action based on current state
        impact = agent.actionToImpact()
        obv, _, _, _ = agent.env.step(impact)
        agent.obvToState(obv)
        if step%(200) == 0:
            agent.interactivePlot()
        agent.env.render()
    agent.env.close()

def main():
    # test()
    # plt.ion()  # 开启interactive mode 成功的关键函数
    # plt.figure(1)
    #
    # for i in range(2000):
    #     plt.clf()  # 清空画布上的所有内容
    #     changingData = np.random.rand(10, 12)
    #     ax = sns.heatmap(changingData)
    #     randX = int(10*np.random.rand(2)[0])
    #     randY = int(10*np.random.rand(2)[1])
    #     rect = patches.Rectangle((randX, randY), 1, 1, linewidth=1, edgecolor='lime', facecolor='lime')
    #     ax.add_patch(rect)
    #     plt.show()
    #     plt.pause(0.01)

    testVar =  1


if __name__ == '__main__':
    mytest = testPointer()
    mytest.equalState()