import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from cliff_env import CliffWalkingEnv

# -----------------------
# تنظیمات
# -----------------------
ALPHA = 0.5
GAMMA = 1.0
EPSILON = 0.1
EPISODES = 500
PLANNING_STEPS = [0, 5, 10]
KAPPA = 0.001   # Exploration bonus coefficient

# -----------------------
# کمکی‌ها
# -----------------------
def state_to_coord(state, env):
    return state // env.cols, state % env.cols

def epsilon_greedy(state, Q, n_actions, env):
    if np.random.rand() < EPSILON:
        return np.random.randint(n_actions)
    row, col = state_to_coord(state, env)
    return np.argmax(Q[row, col])

# -----------------------
# Dyna-Q+
# -----------------------
def dyna_q_plus(n_planning, env, episodes):
    rows, cols = env.rows, env.cols
    n_actions = 4

    Q = np.zeros((rows, cols, n_actions))
    model = {}            # model[state][action] = (reward, next_state)
    tau = np.zeros((rows * cols, n_actions))  # قدم‌های گذشته از آخرین مشاهده

    steps_per_episode = []

    for episode in range(episodes):

        # تغییر محیط در اپیزود 150
        if episode == 150:
            if (4, 10) in env.cliff:
                env.cliff.remove((4, 10))
                print("\n*** Environment changed: (4,10) removed from cliff! ***")

        state = env.reset()
        done = False
        steps = 0

        while not done:

            action = epsilon_greedy(state, Q, n_actions, env)
            next_state, reward, done = env.step(action)

            r, c = state_to_coord(state, env)
            nr, nc = state_to_coord(next_state, env)

            # به‌روز رسانی Q
            best_next = np.argmax(Q[nr, nc])
            td_target = reward + GAMMA * Q[nr, nc, best_next]
            Q[r, c, action] += ALPHA * (td_target - Q[r, c, action])

            # به‌روز رسانی مدل و tau
            tau += 1  # همه مشخصات قدیمی‌تر می‌شوند
            tau[state, action] = 0  # همین state-action تازه استفاده شده
            if state not in model:
                model[state] = {}
            model[state][action] = (reward, next_state)

            # Planning
            for _ in range(n_planning):
                s_rand = random.choice(list(model.keys()))
                a_rand = random.choice(list(model[s_rand].keys()))
                r_sim, s_prime_sim = model[s_rand][a_rand]

                # پاداش اصلاح‌شده با bonus
                bonus = KAPPA * np.sqrt(tau[s_rand, a_rand])
                r_sim_bonus = r_sim + bonus

                sr, sc = state_to_coord(s_rand, env)
                spr, spc = state_to_coord(s_prime_sim, env)
                best = np.argmax(Q[spr, spc])
                td = r_sim_bonus + GAMMA * Q[spr, spc, best]
                Q[sr, sc, a_rand] += ALPHA * (td - Q[sr, sc, a_rand])

            state = next_state
            steps += 1
            if steps > 1000:
                break

        steps_per_episode.append(steps)
    return steps_per_episode


# -----------------------
# اجرای الگوریتم برای n=0,5,10
# -----------------------
env = CliffWalkingEnv()
results = {}

for n in PLANNING_STEPS:
    print(f"\nRunning Dyna-Q+ with n = {n}")
    env = CliffWalkingEnv()
    results[n] = dyna_q_plus(n, env, EPISODES)

# -----------------------
# رسم نمودار
# -----------------------
plt.figure(figsize=(10, 6))

for n, steps in results.items():
    plt.plot(steps, label=f"Dyna-Q+ (n={n})")

plt.title("Dyna-Q+ on Cliff Walking (with Environment Change at Episode 150)")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()
plt.grid(True)
plt.show()
