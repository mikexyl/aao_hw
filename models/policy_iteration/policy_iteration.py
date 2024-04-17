import numpy as np
from models.policy_iteration.policy_iteration_learner import PolicyIteration


def learn(env, scene, max_it, load=None, train=True, **kwargs):
    agent = PolicyIteration(env, scene=scene, gamma=0.95)
    if load is not None:
        agent.load_checkpoint(load)
    if train:
        for epoch in range(max_it):
            print(f'Epoch {epoch}')

            env.reset()
            agent.policy_eval()

            agent.plot.add('mean_value', agent.get_mean_value(), xlabel='epoch', ylabel='mean value',
                           title='Mean Values of ' + agent.algorithm + ' in ' + agent.scene)

            env.reset()
            stable = agent.policy_improvement()

            print('Policy:')
            print(np.round(agent.get_policy().reshape(env.state_shape), 1))

            print('Values:')
            print(np.round(agent.get_values().reshape(env.state_shape), 1))

            if stable:
                print('Policy is stable')
                break

    return agent
