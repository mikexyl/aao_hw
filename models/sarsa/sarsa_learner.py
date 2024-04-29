from models.base_model import ModelFreeAlg
import numpy as np


class SARSA(ModelFreeAlg):
    alg_type = 'SARSA'

    def __init__(self, env, scene, alpha, epsilon, gamma=0.95):
        self.algorithm = self.get_algorithm_name([epsilon, alpha])
        super().__init__(env, scene=scene, algorithm=self.algorithm,
                         alpha=alpha,
                         epsilon=epsilon,
                         gamma=gamma)

    def get_algorithm_name(self, args):
        epsilon, alpha = args[0], args[1]
        return ModelFreeAlg.get_model_free_alg_name([epsilon, alpha, self.alg_type])

    def Q_s_(self, s_, a_):
        return self.get_Q(s_, a_)

    def get_a(self, s, epsilon, alpha=np.array([1, 1, 1, 1])):
        if np.random.random() < epsilon:
            return np.random.randint(self.env.num_actions)
        else:
            return np.argmax(self.get_Q(s) * alpha)
