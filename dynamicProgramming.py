# represents the DP agent

import helper

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
    return None


if __name__ == '__main__':
    main()