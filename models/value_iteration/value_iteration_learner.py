import numpy as np
from models.base_model import ModelBasedAlg


class ValueIteration(ModelBasedAlg):
    def __init__(self, env, scene, gamma=0.95, theta=1e-9, max_it=1000):
        self.algorithm = self.get_algorithm_name([])
        super().__init__(env, scene=scene, algorithm=self.algorithm, gamma=gamma, theta=theta, max_it=max_it)

        # re-initialize values and policy to support discrete belief
        self.res = 0.01
        self.bins = int(1 / self.res)
        self.num_states = env.num_states * self.bins
        self._values = np.random.rand(self.num_states)
        # generate random policy with uniform distribution sum to 1
        self._policy = np.random.randint(
            low=0, high=self.env.num_actions, size=self._values.shape)

    def belief_to_state(self, b):
        return int(b[0] / self.res)

    def state_to_belief(self, s):
        return [(s + 0.5) * self.res, 1 - (s + 0.5) * self.res]

    def get_algorithm_name(self, args):
        return 'value_iteration'

    def value_iteration(self):
        i = 0
        delta = 0
        for s in range(self.env.num_states):
            max_a = -1
            max_v = -np.inf
            old_v = self.get_values(s)
            for a in range(self.env.num_actions):
                _, new_value = self.eval_state(s, a)
                if new_value > max_v:
                    max_v = new_value
                    max_a = a
            self.set_values(s, max_v)
            self.set_policy(s, max_a)
            delta = max(delta, abs(old_v - max_v))

        print(f'delta: {delta}')

        if delta < self.theta:
            return True, delta
        else:
            return False, delta

    def value_iteration_belief(self):
        i = 0
        delta = 0
        for sb in range(self.num_states):
            b = self.state_to_belief(sb)
            max_a = -1
            max_v = -np.inf
            old_v = self.get_values(sb)
            for a in range(self.env.num_actions):
                new_value_action = 0
                for real_s in range(self.env.num_states):
                    _, new_value = self.eval_belief(real_s, b, a)
                    new_value_action += new_value * b[real_s]
                if new_value_action > max_v:
                    max_v = new_value_action
                    max_a = a
            self.set_values(sb, max_v)
            self.set_policy(sb, max_a)
            delta = max(delta, abs(old_v - max_v))

        print(f'delta: {delta}')

        if delta < self.theta:
            return True, delta
        else:
            return False, delta
