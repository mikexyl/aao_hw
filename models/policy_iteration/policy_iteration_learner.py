import numpy as np
from models.base_model import ModelBasedAlg


class PolicyIteration(ModelBasedAlg):
    def __init__(self, env, scene, gamma=0.95, theta=1e-9, max_it=100):
        self.algorithm = self.get_algorithm_name([])
        super().__init__(env, scene, algorithm=self.algorithm, gamma=gamma, theta=theta, max_it=max_it)
        self.gamma = gamma
        self.theta = theta
        self.max_it = max_it

    def get_algorithm_name(self, args):
        return 'policy_iteration'

    def policy_eval(self):
        i = 0
        delta = 0
        for i in range(self.max_it):
            delta = 0
            # iterate all states
            for s in range(self.env.num_states):
                new_delta, new_value = self.eval_state(s)
                delta = max(new_delta, delta)
                self.set_values(s, new_value)

            # add to log

            # calculate the difference between new value and old value
            # if the difference is smaller than theta, stop the iteration
            if delta < self.theta:
                break

        print(f'policy evaluation finished after {i + 1} iterations with delta= {delta}')

    def policy_improvement(self):
        policy_stable = True
        old_policy = self.get_policy().copy()

        for s in range(self.env.num_states):
            max_a = -1
            max_v = -np.inf
            # iterate all next states
            for a in range(self.env.num_actions):
                a_value = 0
                for s_ in range(self.env.num_states):
                    a_value += self.env.p(s_, s, a) * (self.env.r(s, a) + self.gamma * self.get_values(s_))
                if a_value > max_v:
                    max_v = a_value
                    max_a = a

            assert max_a != -1, 'max_a should not be -1'
            self.set_policy(s, max_a)

        policy_delta = np.linalg.norm(self.get_policy() - old_policy)

        if policy_delta != 0:
            policy_stable = False
        print(f'policy improvement finished with policy delta: {policy_delta}')

        return policy_stable
