import imageio

import numpy as np
np.set_printoptions(threshold=np.inf)

from make_env import make_env
env = make_env()

Gt = 0.0
reward = 0.0
state = env.reset()
done = False
t = 0
frames = []
while not done:
    # env.render(mode="full")
    env.render()

    action = input("input action: ")
    while action == "":
        action = input("input action: ")

    while int(action)>4:
        action = input("input action: ")
    action = int(action)

    state_curr, reward, done, _info = env.step(action)

    print(f"Reward={reward}")

    state = state_curr
    Gt += reward
    t += 1

print(f"Finished at timestep:{t} with Total reward:{Gt}")
imageio.mimwrite(f'MiniHack-Quest-Hard-v0_human_gameplay.gif', frames, fps=10)

env.close()
