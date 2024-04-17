from models.sarsa.sarsa_learner import SARSA
import numpy as np


def learn(env, scene, max_it, epsilon, alpha, **kwargs):
    agent = SARSA(env, scene, epsilon=epsilon, alpha=alpha)

    for episode in range(max_it):
        # print(f'Epoch {episode}')
        env.reset()
        done = False

        a = agent.get_a(env.s, agent.epsilon)
        return_per_episode = 0
        while not done:
            s = env.s
            s_, r, done = env.step(a)
            a_ = agent.get_a(s_, agent.epsilon)
            agent.update_Q(s, a, r, s_, a_, done)
            a = a_

            return_per_episode += r

        # print(agent.Q)
        # print(np.linalg.norm(agent.Q-old_q))
        agent.plot.add('return_per_episode', return_per_episode, xlabel='episode', ylabel='return',
                       title='Return per Episode of ' + agent.algorithm + ' in ' + agent.scene)

    # print(agent.Q)

    # print(agent.get_policy_for_all_s())

    return agent
