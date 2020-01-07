from old import test
import dynamicProgramming
import deepQLearning
import QLearning

# in the main we are going to run different methodes and do the comparisons.

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

if __name__ == '__main__':
    print("main")
    test.main()
    print(test.testVar)