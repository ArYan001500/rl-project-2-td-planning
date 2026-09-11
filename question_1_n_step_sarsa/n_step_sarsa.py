import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from cliff_env import CliffWalkingEnv

# ایجاد محیط
env = CliffWalkingEnv()

# پارامترها
alpha = 0.5
gamma = 1.0
epsilon = 0.1

episodes = 500

rows = env.rows
cols = env.cols
n_actions = 4

# ساخت Q-table
Q = np.zeros((rows, cols, n_actions))


# سیاست epsilon-greedy
def epsilon_greedy(state):

    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)

    r, c = state
    return np.argmax(Q[r, c])

def n_step_sarsa(n):

    steps_per_episode = []

    for episode in range(episodes):

        state = env.reset()
        action = epsilon_greedy(state)

        # لیست‌ها برای ذخیره
        states = [state]
        actions = [action]
        rewards = [0]   # پاداش اولیه صفر

        T = float('inf')
        t = 0

        while True:

            if t < T:

                next_state, reward, done = env.step(action)
                rewards.append(reward)
                states.append(next_state)

                if done:
                    T = t + 1
                else:
                    next_action = epsilon_greedy(next_state)
                    actions.append(next_action)

            tau = t - n + 1

            if tau >= 0:

                # محاسبه G (بازگشت n-step)
                G = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += rewards[i]

                if tau + n < T:
                    r, c = states[tau + n]
                    G += Q[r, c, actions[tau + n]]

                # به‌روزرسانی Q
                r0, c0 = states[tau]
                a0 = actions[tau]

                Q[r0, c0, a0] += alpha * (G - Q[r0, c0, a0])

            if tau == T - 1:
                break

            t += 1
            if t < T:
                action = actions[t]

        steps_per_episode.append(t)

    return steps_per_episode

if __name__ == "__main__":
    steps_1 = n_step_sarsa(1)
    steps_5 = n_step_sarsa(5)
    steps_10 = n_step_sarsa(10)

    print("Training finished.")

episodes_range = range(episodes)

plt.figure(figsize=(8, 5))
plt.plot(episodes_range, steps_1, label="n = 1")
plt.plot(episodes_range, steps_5, label="n = 5")
plt.plot(episodes_range, steps_10, label="n = 10")

plt.xlabel("Episode")
plt.ylabel("Steps per Episode")
plt.title("n-step Sarsa on 5x20 Cliff Walking")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("sarsa_cliff_results.png")
plt.show()


