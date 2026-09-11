import numpy as np


class CliffWalkingEnv:

    def __init__(self):

        self.rows = 5
        self.cols = 20

        self.start = (4, 0)
        self.goal = (4, 19)

        self.cliff = [(4, i) for i in range(1, 19)]

        self.actions = [
            (-1, 0),   # up
            (1, 0),    # down
            (0, -1),   # left
            (0, 1)     # right
        ]

        self.reset()

    def state_to_index(self, state):
        r, c = state
        return r * self.cols + c

    def reset(self):

        self.agent_pos = self.start
        return self.state_to_index(self.agent_pos)

    def step(self, action):

        move = self.actions[action]

        r = self.agent_pos[0] + move[0]
        c = self.agent_pos[1] + move[1]

        r = max(0, min(self.rows - 1, r))
        c = max(0, min(self.cols - 1, c))

        next_state = (r, c)

        reward = -1
        done = False

        if next_state in self.cliff:

            reward = -100
            next_state = self.start

        if next_state == self.goal:

            done = True

        self.agent_pos = next_state

        return self.state_to_index(next_state), reward, done
