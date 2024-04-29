import random
import numpy as np

# altitude is a 7 by 7 grid
altitude_lut = [[5, 3, 4, 2, 0, 0, 4],
                [5, -1, 4, 2, 4, 5, 5],
                [0, 4, 3, 5, 3, 1, 2],
                [2, 0, 0, 0, 5, 1, 5],
                [0, 2, 3, 5, 3, 2, 3],
                [1, 3, -2, 5, 4, 0, 5],
                [4, 2, 2, 3, 4, 4, 4]]

RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3


class MountainGridWorld():
    """
    The world is a 5 x 5 grid based on Example 3.5 from Sutton 2019. There are 25 states. We index these states as follows:

        0   1   2   3   4
        5   6   7   8   9
        10  11  12  13  14
        15  16  17  18  19
        20  21  22  23  24

    For example, state "1" is cell "A" in Sutton 2019, state "3" is cell "B", and so forth.

    There are 4 actions. We index these actions as follows:

                1 (up)
        2 (left)        0 (right)
                3 (down)

    If you specify hard_version=True, then the action will be selected uniformly at random 10% of the time.
    """

    def __init__(self, perfect_obs=False, max_num_steps=1000):
        self.num_states = 49
        self.num_actions = 4
        self.last_action = None
        self.max_num_steps = max_num_steps
        self.state_shape = (7, 7)
        self.state_names = ['x', 'y']
        self.start_state = 8
        self.end_state = 37
        self.perfect_obs = perfect_obs
        self.end_pos = self.get_pos(self.end_state)
        self.reset()

    def altitude(self, s):
        # s must be one integer smaller than 7*7
        assert s < self.num_states

        x, y = self.get_pos(s)
        altitude = altitude_lut[x][y]
        assert altitude >= 0

    def get_pos(self, s):
        assert self.state_shape[0] == self.state_shape[1]
        return s // self.state_shape[0], s % self.state_shape[0]

    def p(self, s1, s, a):
        raise NotImplementedError

    def r(self, s):
        if s == self.end_state:
            return 100  # goal reward
        elif s < 0 or s >= 49:
            return -10
        else:
            pos = self.get_pos(s)
            # return the distance to the goal
            return -np.sqrt(abs(pos[0] - self.end_pos[0])**2 + abs(pos[1] - self.end_pos[1])**2)

    def step(self, a, verbose=False):
        last_s = self.s

        # real s1
        s1, success = self.apply_action_with_uncertainty(self.s, a)
        r = self.r(s1)
        if r == -10:
            s1 = last_s
        done = self.is_done(s1)
        # print("from state", self.s, "action", a, "to state", s1, "reward", r, "done", done)

        if done:
            s1 = self.s
            obs = s1
            confidence = 1
        elif self.perfect_obs:
            self.s = s1
        else:
            # noisy obs
            obs, confidence = self.get_obs(s1)
            self.s = s1
        self.num_steps += 1

        if verbose:
            print(f"from state {last_s} action {a} to state {s1} reward {r} done {done} confidence {confidence}")

        # if done:
        #     print(f"done at {self.num_steps} steps")
        if self.perfect_obs:
            return s1, r, done, success
        else:
            return s1, r, done, obs, confidence

    def get_next_obs_and_confidence(self, s1, obs, a):
        s1_obs = self.apply_action(obs, a)
        confidence = self.get_confidence(s1)
        return s1_obs, confidence

    def get_confidence(self, s):
        x, y = self.get_pos(s)
        altitude = altitude_lut[x][y]
        return 0.5 + 0.09 * altitude

    def is_done(self, s):
        return s == self.end_state or self.num_steps >= self.max_num_steps or s < 0 or s >= 49

    def get_possible_actions(self, s, a=None):
        # if s is in border
        x, y = self.get_pos(s)
        actions = None
        if y == 0:
            if x == 0:
                actions = [RIGHT, DOWN]
            elif x == 6:
                actions = [RIGHT, UP]
            else:
                actions = [RIGHT, UP, DOWN]
        elif y == 6:
            if x == 0:
                actions = [LEFT, DOWN]
            elif x == 6:
                actions = [LEFT, UP]
            else:
                actions = [LEFT, UP, DOWN]
        elif x == 0:
            actions = [RIGHT, LEFT, DOWN]
        elif x == 6:
            actions = [RIGHT, LEFT, UP]
        else:
            actions = [RIGHT, LEFT, UP, DOWN]

        assert actions is not None, f"actions is None for state {s}, {x}, {y}"
        if a is not None and a in actions:
            # remove a from actions
            actions.remove(a)

        return actions

    @staticmethod
    def apply_action(s, a):
        if a == RIGHT:
            return s + 1
        if a == UP:
            return s - 7
        if a == LEFT:
            return s - 1
        if a == DOWN:
            return s + 7
        return s

    def apply_action_with_uncertainty(self, s, a):
        accurate_p = 0.75
        possible_actions = self.get_possible_actions(s)
        # if move is impossible, move the state to -1 for penalty
        if a not in possible_actions:
            return -1, True
        possible_actions.remove(a)
        if random.uniform(0, 1) < accurate_p:
            # accurate
            return self.apply_action(s, a), True
        else:
            random_action = random.choice(self.get_possible_actions(s, a))
            return self.apply_action(s, random_action), False

    def get_possible_wrong_obs(self, s):
        possible_actions = self.get_possible_actions(s)
        possible_s = []
        for a in possible_actions:
            obs = self.apply_action(s, a)
            assert 0 <= obs < 49, f"Invalid state {s}, {a}, {obs}, {self.get_pos(s)}"
            possible_s.append(obs)
        return possible_s

    def get_possible_wrong_r(self, s):
        possible_wrong_s = self.get_possible_wrong_obs(s)
        possible_wrong_r = []
        for s in possible_wrong_s:
            possible_wrong_r.append(self.r(s))
        return possible_wrong_r

    def get_obs(self, s):
        x, y = self.get_pos(s)
        assert self.state_shape[0] > x >= 0 and self.state_shape[1] > y >= 0, f"Invalid state {s}, {x}, {y}"
        altitude = altitude_lut[x][y]
        if altitude < 0:
            return s, 1

        obs_p = 0.5 + 0.09 * altitude
        obs = random.uniform(0, 1)
        if obs < obs_p:
            return s, obs_p
        else:
            possible_s = self.get_possible_wrong_obs(s)
            # TODO: check if choice is uniform
            return random.choice(possible_s), obs_p

    def reset(self):
        # Choose initial state uniformly at random
        self.s = self.start_state
        self.num_steps = 0
        self.last_action = None
        return self.s

    def render(self):
        k = 0
        output = ''
        for i in range(5):
            for j in range(5):
                if k == self.s:
                    output += 'X'
                elif k == 1 or k == 3:
                    output += 'o'
                else:
                    output += '.'
                k += 1
            output += '\n'
        if self.last_action is not None:
            print(['right', 'up', 'left', 'down'][self.last_action])
            print()
        print(output)
