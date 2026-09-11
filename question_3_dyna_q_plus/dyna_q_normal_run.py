import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import random
from cliff_env import CliffWalkingEnv

# -----------------------------
#  تنظیمات اولیه و هایپرپارامترها
# -----------------------------
ALPHA = 0.5
GAMMA = 1.0
EPSILON = 0.1
EPISODES = 500
PLANNING_STEPS = [0, 5, 10]


# -----------------------------
# تبدیل state عددی → مختصات (row, col)
# -----------------------------
def state_to_coord(state, env):
    row = state // env.cols
    col = state % env.cols
    return (row, col)


# -----------------------------
#  سیاست ε-greedy
# -----------------------------
def epsilon_greedy(state, Q, n_actions, env):
    row, col = state_to_coord(state, env)

    if np.random.rand() < EPSILON:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[row, col])


# -----------------------------
#  الگوریتم DYNA-Q
# -----------------------------
def dyna_q(n_planning, env, episodes):
    rows, cols = env.rows, env.cols
    n_actions = 4

    # جدول Q
    Q = np.zeros((rows, cols, n_actions))

    # مدل محیط
    model = {}  # model[(s)][a] = (r, s')

    steps_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done:

            #  انتخاب اکشن
            action = epsilon_greedy(state, Q, n_actions, env)

            #  اجرای اکشن
            next_state, reward, done = env.step(action)

            # مختصات state و next_state
            row, col = state_to_coord(state, env)
            next_row, next_col = state_to_coord(next_state, env)


            best_next_action = np.argmax(Q[next_row, next_col])
            td_target = reward + GAMMA * Q[next_row, next_col, best_next_action]
            Q[row, col, action] += ALPHA * (td_target - Q[row, col, action])


            if state not in model:
                model[state] = {}

            model[state][action] = (reward, next_state)


            for _ in range(n_planning):
                s_rand = random.choice(list(model.keys()))
                a_rand = random.choice(list(model[s_rand].keys()))
                r_sim, s_prime_sim = model[s_rand][a_rand]

                # تبدیل حالت شبیه‌سازی‌شده
                s_r, s_c = state_to_coord(s_rand, env)
                sp_r, sp_c = state_to_coord(s_prime_sim, env)

                best_sim_action = np.argmax(Q[sp_r, sp_c])
                td_target_sim = r_sim + GAMMA * Q[sp_r, sp_c, best_sim_action]
                Q[s_r, s_c, a_rand] += ALPHA * (td_target_sim - Q[s_r, s_c, a_rand])

            # ادامه مسیر
            state = next_state
            steps += 1

            if steps > 1000:
                break

        steps_per_episode.append(steps)

    return steps_per_episode


# -----------------------------
# اجرای Dyna-Q برای n = 0, 5, 10
# -----------------------------
env = CliffWalkingEnv()
results = {}

for n in PLANNING_STEPS:
    print(f"\nRunning Dyna-Q with n = {n}")
    results[n] = dyna_q(n, env, EPISODES)


# رسم نمودار

plt.figure(figsize=(10, 6))

for n, steps in results.items():
    plt.plot(steps, label=f"Dyna-Q (n={n})")

plt.title("Dyna-Q on Cliff Walking (Steps per Episode)")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()
plt.grid(True)
plt.show()
