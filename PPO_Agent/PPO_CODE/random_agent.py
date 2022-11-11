import imageio

from make_env import make_env

env = make_env()

Gt = 0.0
frames=[]
state = env.reset()
for t in range(1000):
        env.render()
        frame = state["pixel"]
        frames.append(frame)
        action = env.action_space.sample() # random action selected from action space
        state, reward, done, _info = env.step(action)
        Gt += reward
        if done: 
            print(f"Finished at timestep:{t} with Total reward:{Gt}")
            break
imageio.mimwrite(f'MiniHack-Quest-Hard-v0_random_agent.gif', frames, fps=10)  
env.close()

