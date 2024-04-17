import random


class GoodBad(object):
    def __init__(self, max_num_steps=100):
        self.max_num_steps = max_num_steps
        self.num_states = 2  # 0: good, 1: bad
        self.num_actions = 3  # 0: switch, 1: stay, 2: locate
        self.state_shape = (2,)
        self.state_name = ['good/bad']
        self.s = 0  # start from good state
        self.num_steps = 0
        self.last_action = None

    def reset(self):
        self.s = random.randint(0, 1)
        self.num_steps = 0
        self.last_action = None
        return self.s

    def get_pos(self, s):
        return s

    def p(self, s1, s, a):
        if a == 0:  # switch
            if s1 == s:  # switch but stay
                return 0
            else:  # switch and switch
                return 1
        if a == 1:  # stay action
            if s1 == s:  # stay and stayed
                return 0.99
            else:  # stay but switched
                return 0.01
        if a == 2:  # locate action
            if s1 == s:  # locate and stayed
                return 1
            else:  # locate but switched
                return 0

    def r(self, s, a):
        if s == 0 and a == 1:  # R(good, stay)
            return 5
        if s == 1 and a == 1:  # R(bad, stay)
            return -5
        if s == 0 and a == 2:  # R(good, locate)
            return -1
        if s == 1 and a == 2:  # R(bad, locate)
            return -1
        return 0

    def step(self, a):
        self.last_action = a

        old_s= self.s
        r = self.r(self.s, a)

        # uniform random transition
        prob = random.uniform(0, 1)
        res_prob = 0
        for s1 in range(self.num_states):
            if prob < self.p(s1, self.s, a) + res_prob:
                self.s = s1
                break
            else:
                res_prob += self.p(s1, self.s, a)

        self.num_steps += 1

        return self.s, r, self.num_steps >= self.max_num_steps
