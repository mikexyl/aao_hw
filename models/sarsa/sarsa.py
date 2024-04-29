from models.sarsa.sarsa_learner import SARSA
import numpy as np


def learn(env, scene, max_it, epsilon, alpha, epsilon_decay=0.99, **kwargs):
    agent = SARSA(env, scene, epsilon=epsilon, alpha=alpha)

    for episode in range(max_it):
        # print(f'Epoch {episode}')
        env.reset()
        done = False

        a = agent.get_a(env.s, agent.epsilon)
        return_per_episode = 0
        alpha_hat = np.array([1.0, 1.0, 1.0, 1.0])
        n_a = np.zeros(4)
        n_a_a = np.zeros(4)
        while not done:
            s = env.s
            s_, r, done, success = env.step(a)
            alpha_hat_ = np.array([1.0, 1.0, 1.0, 1.0])
            if np.all(alpha_hat >0):
                alpha_hat_ = alpha_hat
                epsilon = agent.epsilon
                agent.epsilon *= epsilon_decay
            else:
                alpha_hat_ = np.array([1.0, 1.0, 1.0, 1.0])
                epsilon = 1.0

            # a_ = agent.get_a(s_, epsilon, alpha_hat_)
            a_ = agent.get_a(s_, epsilon, np.array([1.0, 1.0, 1.0, 1.0]))

            n_a[a] += 1
            if success:
                n_a_a[a] += 1
            if n_a[a] > 0:
                alpha_hat[a] = float(n_a_a[a] / n_a[a])

            agent.update_Q(s, a, r*alpha_hat_[a], s_, a_, done)

            a = a_

            return_per_episode += r

        print(f'Episode {episode} return: {return_per_episode} done in {env.num_steps} steps')

        # print(agent.Q)
        # print(np.linalg.norm(agent.Q-old_q))
        # agent.plot.add('return_per_episode', return_per_episode, xlabel='episode', ylabel='return',
        #                title='Return per Episode of ' + agent.algorithm + ' in ' + agent.scene)

    # print(agent.Q)

    # print(agent.get_policy_for_all_s())

    return agent, alpha_hat
