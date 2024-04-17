import numpy as np
from models.value_iteration.value_iteration_learner import ValueIteration


def learn(env, scene, max_it, theta=1e-09, load=None, **kwargs):
    agent = ValueIteration(env, scene=scene, theta=theta)

    if load is not None:
        agent.load_checkpoint(load)
    else:
        for epoch in range(max_it):
            reached_theta, delta = agent.value_iteration()

            agent.plot.add('mean_value', agent.get_mean_value(), xlabel='epoch', ylabel='mean value',
                           title='Mean Values of ' + agent.algorithm + ' in ' + agent.scene)

            print('Learning finished after {} epochs, converged {}, delta {}'.format(epoch, reached_theta, delta))

            if reached_theta:
                break
    return agent
