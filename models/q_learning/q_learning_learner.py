from models.base_model import ModelFreeAlg
import numpy as np


class QLearning(ModelFreeAlg):
    alg_type = 'Q-learning'

    def __init__(self, env, scene, alpha, epsilon, gamma=0.95, num_states=None):
        self.algorithm = self.get_algorithm_name([epsilon, alpha])
        super().__init__(env, scene=scene,
                         algorithm=self.algorithm, alpha=alpha,
                         epsilon=epsilon,
                         gamma=gamma)
        if num_states is not None:
            self.Q = np.random.rand(num_states, self.env.num_actions)

    def get_algorithm_name(self, args):
        epsilon, alpha = args[0], args[1]
        return ModelFreeAlg.get_model_free_alg_name([epsilon, alpha, self.alg_type])

    def Q_s_(self, s_, a_):
        assert a_ is None, 'Q-learning does not use a_'
        return np.max(self.get_Q(s_))
