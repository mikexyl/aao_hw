import test


def main():
    algorithms_for_scenes = {
        # 'gridworld': ['sarsa', 'q_learning', 'policy_iteration', 'value_iteration']}
        'pendulum': ['sarsa', 'q_learning']}
    max_it_n = {'sarsa': 2000, 'q_learning': 2000, 'policy_iteration': 200, 'value_iteration': 200}
    epsilon_fixed = 0.2
    alpha_fixed = 0.7
    epsilon_n = [0.1, 0.2, 0.3, 0.4, epsilon_fixed, epsilon_fixed, epsilon_fixed, epsilon_fixed]
    alpha_n = [alpha_fixed, alpha_fixed, alpha_fixed, alpha_fixed, 0.2, 0.4, 0.6, 0.8]

    test.run(algorithms_for_scenes, epsilon_n, alpha_n, max_it_n)


if __name__ == '__main__':
    main()
