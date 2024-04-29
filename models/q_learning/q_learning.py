from models.q_learning.q_learning_learner import QLearning
import numpy as np

PRECISE = 0
NOISY = 1

DISCRETIZATION_RESOLUTION = 10


def discretize_state_and_confidence(obs, confidence):
    # convert confidence into 0 to 9
    confidence_bin = int(round(confidence * (DISCRETIZATION_RESOLUTION - 1)))
    assert 0 <= confidence_bin <= 9
    v = obs * DISCRETIZATION_RESOLUTION + confidence_bin
    assert 0 <= v <= 489, f"obs {obs}, confidence {confidence}, confidence_bin {confidence_bin}, v {v}"
    return v


def learn(env, scene, max_it, epsilon, alpha, epsilon_decay, **kwargs):
    num_state = None

    agent = QLearning(env, scene=scene, epsilon=epsilon, alpha=alpha, num_states=num_state)

    for episode in range(int(max_it)):
        # Initialize S
        env.reset()
        done = False

        return_per_episode = 0
        alpha_hat = np.array([0.0, 0.0, 0.0, 0.0])
        n_a = np.zeros(4)
        n_a_a = np.zeros(4)
        while not done:
            min_a = None
            if np.all(alpha_hat > 0):
                agent.epsilon *= epsilon_decay
            else:
                alpha_hat_ = np.array([1.0, 1.0, 1.0, 1.0])
                # pick which n_a_a is smallest
                min_a = np.argmin(alpha_hat)

            s = env.s
            if min_a is not None:
                a = min_a
            else:
                a = agent.get_a(env.s, agent.epsilon, np.array([1.0, 1.0, 1.0, 1.0]))
            s_, r, done, success = env.step(a)

            n_a[a] += 1
            if success:
                n_a_a[a] += 1
            if n_a_a[a] > 3:
                alpha_hat[a] = float(n_a_a[a] / n_a[a])

            discount = n_a[a] / 5
            if discount > 1:
                discount = 1

            agent.update_Q(s, a, r * alpha_hat[a] * discount, s_, None, done)

        # print(agent.Q)
        # print(np.linalg.norm(agent.Q-old_q))
        # agent.plot.add('return_per_episode', return_per_episode, xlabel='episode', ylabel='return',
        #                title='Return per Episode of ' + agent.algorithm + ' in ' + agent.scene)
        #
        if episode % (0.1 * max_it) == 0:
            print(f'Episode {episode} of {max_it} finished, steps {env.num_steps}')

    return agent
