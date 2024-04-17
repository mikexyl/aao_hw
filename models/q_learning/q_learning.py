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


def learn(env, scene, max_it, epsilon, alpha, obs_mode, **kwargs):
    num_state = None
    if obs_mode == PRECISE:
        num_state = env.num_states
    elif obs_mode == NOISY:
        num_state = env.num_states * DISCRETIZATION_RESOLUTION
    agent = QLearning(env, scene=scene, epsilon=epsilon, alpha=alpha, num_states=num_state)

    for episode in range(int(max_it)):
        # Initialize S
        env.reset()
        done = False

        return_per_episode = 0
        while not done:
            s = env.s
            # Choose A from S using episilon-greedy policy
            a = agent.get_a(env.s, agent.epsilon)
            # Take A, observe R, S'
            results = env.step(a)
            if len(results) == 3:
                s1, r, done = results
            elif len(results) == 5:
                s1, r, done, s1_obs, confidence = results
            else:
                raise ValueError(f"results has length {len(results)}")

            if obs_mode == PRECISE:
                agent.update_Q(s, a, r, s1, None, done)
            elif obs_mode == NOISY:
                s_obs, s_confidence = env.get_obs(s)
                assert 0 <= s1_obs << env.num_states
                assert 0 <= s_obs << env.num_states
                # get possible r of wrong_s
                wrong_r = env.get_possible_wrong_r(s1)
                r_biased = r + sum(wrong_r) * (1 - confidence)

                s1_obs_and_c = discretize_state_and_confidence(s1_obs, confidence)
                s_obs_and_c = discretize_state_and_confidence(s_obs, s_confidence)
                agent.update_Q(s_obs_and_c, a, r_biased, s1_obs_and_c, None, done)
            else:
                agent.update_Q(s, a, r, s1, None, done)

            return_per_episode += r

        # print(agent.Q)
        # print(np.linalg.norm(agent.Q-old_q))
        # agent.plot.add('return_per_episode', return_per_episode, xlabel='episode', ylabel='return',
        #                title='Return per Episode of ' + agent.algorithm + ' in ' + agent.scene)
        #
        if episode % (0.1 * max_it) == 0:
            print(f'Episode {episode} of {max_it} finished')

    return agent
