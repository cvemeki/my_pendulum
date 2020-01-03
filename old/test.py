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

    testVar = testVar + 1


if __name__ == '__main__':
    global testVar
    testVar = 1
    main()