import numpy as np

class CliffWalkingEnv:

    def __init__(self, rows=5, cols=20):
        self.rows = rows
        self.cols = cols

        # موقعیت شروع و هدف
        self.start = (rows-1, 0)
        self.goal = (rows-1, cols-1)

        # موقعیت عامل
        self.state = self.start

        # تعریف ناحیه cliff
        self.cliff = [(rows-1, c) for c in range(1, cols-1)]

        # اکشن ها: بالا، پایین، چپ، راست
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):

        move = self.actions[action]

        next_row = self.state[0] + move[0]
        next_col = self.state[1] + move[1]

        # جلوگیری از خروج از محیط
        next_row = max(0, min(self.rows-1, next_row))
        next_col = max(0, min(self.cols-1, next_col))

        next_state = (next_row, next_col)

        reward = -1
        done = False

        # اگر داخل cliff بیفتد
        if next_state in self.cliff:
            reward = -100
            next_state = self.start

        # اگر به هدف برسد
        if next_state == self.goal:
            done = True

        self.state = next_state

        return next_state, reward, done
