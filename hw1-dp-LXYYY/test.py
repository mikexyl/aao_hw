import matplotlib.pyplot as plt
import gridworld
import discrete_pendulum
from utils import get_learn_function
from models import Plot
import numpy as np


def test_x_to_s(env):
    theta = np.linspace(-np.pi * (1 - (1 / env.n_theta)), np.pi * (1 - (1 / env.n_theta)), env.n_theta)
    thetadot = np.linspace(-env.max_thetadot * (1 - (1 / env.n_thetadot)),
                           env.max_thetadot * (1 - (1 / env.n_thetadot)), env.n_thetadot)
    for s in range(env.num_states):
        i = s // env.n_thetadot
        j = s % env.n_thetadot
        s1 = env._x_to_s([theta[i], thetadot[j]])
        if s1 != s:
            raise Exception(f'test_x_to_s: error in state representation: {s} and {s1} should be the same')
    print('test_x_to_s: passed')


def run(algorithms_for_scenes, epsilon_n, alpha_n, max_it_n):
    load_checkpoint = False
    train = True

    for scene, algorithms in algorithms_for_scenes.items():
        # Create environment
        env = gridworld.GridWorld(hard_version=False) if scene == 'gridworld' else discrete_pendulum.Pendulum(
            n_theta=31,
            n_thetadot=31)

        if scene == 'pendulum':
            test_x_to_s(env)

        for alg in algorithms:
            max_it = max_it_n[alg]
            if alg == 'sarsa' or alg == 'q_learning':
                epsilon = epsilon_n
                alpha = alpha_n
            else:
                epsilon = [0]
                alpha = [0]

            for eps, alp in zip(epsilon, alpha):
                env.reset()
                learn = get_learn_function(alg=alg)
                checkpoint = 'ckp' if load_checkpoint else None
                model = learn(env, scene=scene, epsilon=eps, alpha=alp, max_it=max_it, load=checkpoint, train=train)

                # model = learn(env, scene='pendulum', epsilon=0.1, alpha=0.5, max_it=2000)
                # model.save_checkpoint('ckp')

                # Initialize simulation
                s = env.reset()

                # Create log to store data from simulation
                if scene == 'gridworld':
                    log = {
                        't': [0],
                        's': [s],
                        'a': [],
                        'r': [],
                    }
                else:
                    log = {
                        't': [0],
                        's': [s],
                        'a': [],
                        'r': [],
                        'theta': [env.x[0]],  # agent does not have access to this, but helpful for display
                        'thetadot': [env.x[1]],  # agent does not have access to this, but helpful for display
                    }

                # Simulate until episode is done
                done = False

                while not done:
                    a = model.get_policy(s)
                    (s, r, done) = env.step(a)
                    log['t'].append(log['t'][-1] + 1)
                    log['s'].append(s)
                    log['a'].append(a)
                    log['r'].append(r)
                    if scene == 'pendulum':
                        log['theta'].append(env.x[0])
                        log['thetadot'].append(env.x[1])

                    model.plot.add('trajectory', env.get_pos(s), 'trajectory', alpha=0.5,
                                   title='Trajectory of ' + model.algorithm + ' in ' + scene, xlabel='step', ylabel='l')

                model.plot.plot_policy(model.get_policy(), save=True)
                model.plot.plot_state_value_function(model.get_state_value_function(), save=True)
                model.plot.plot(save=True)

                if scene == 'gridworld':
                    # Plot data and save to png file
                    plt.plot(log['t'], log['s'])
                    plt.plot(log['t'][:-1], log['a'])
                    plt.plot(log['t'][:-1], log['r'])
                    plt.legend(['s', 'a', 'r'])
                    plt.savefig('figures/gridworld/log_' + model.algorithm + '.png')
                    plt.show()
                else:
                    # Plot data and save to png file
                    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
                    ax[0].plot(log['t'], log['s'])
                    ax[0].plot(log['t'][:-1], log['a'])
                    ax[0].plot(log['t'][:-1], np.array(log['r']) * 200)
                    ax[0].legend(['s', 'a', 'r'])
                    ax[1].plot(log['t'], log['theta'])
                    ax[1].plot(log['t'], log['thetadot'])
                    ax[1].legend(['theta', 'thetadot'])
                    plt.savefig('figures/pendulum/log_' + model.algorithm + '.png')
                    plt.show()

                print('Total reward: ', sum(log['r']))

        algorithms = []
        n_epsilon = 4

        from models import ModelFreeAlg
        from models import SARSA
        from models import QLearning

        for alg in [SARSA.alg_type, QLearning.alg_type]:
            algorithms = []
            for epsilon, alpha in zip(epsilon_n, alpha_n):
                algorithms.append(ModelFreeAlg.get_model_free_alg_name([epsilon, alpha, alg]))
            Plot.plot_compare([scene for _ in range(n_epsilon)], algorithms[:n_epsilon], key='return_per_episode',
                              title=r'Return per episode for different $\epsilon$ in ' + scene + ' of ' + alg,
                              save=True,
                              plot_interval=False)
            Plot.plot_compare([scene for _ in range(len(algorithms) - n_epsilon)], algorithms[n_epsilon:],
                              key='return_per_episode',
                              title=r'Return per episode for different $\alpha$ in ' + scene + ' of ' + alg, save=True,
                              plot_interval=False)

        Plot.save_all_plots('figures/data')
