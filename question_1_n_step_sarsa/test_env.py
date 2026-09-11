from cliff_env import CliffWalkingEnv

env = CliffWalkingEnv()

state = env.reset()
print("Start State:", state)

# چند حرکت به راست انجام می‌دهیم
for i in range(5):

    action = 3   # 3 یعنی حرکت به راست

    next_state, reward, done = env.step(action)

    print("Next State:", next_state, "Reward:", reward)

    if done:
        print("Goal reached!")
        break
