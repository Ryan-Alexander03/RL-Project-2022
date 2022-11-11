import numpy as np

from make_env import make_env

from stable_baselines3 import PPO

import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw

env = make_env()

def label_with_episode_number(frame, ep, timestep, running_rewards):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)

    drawer.text((im.size[0]/20,im.size[1]/20), f'Episode: {ep}\tTimestep: {timestep}\nRunning Reward:{running_rewards}', fill=text_color)

    return im

def render_model(model, env, num_ep):
    for ep in range(num_ep):
        frames=[]
        done = False
        running_rewards = 0
        obs = env.reset()

        time_step = 0
        while not done:
            env.render()
            frame = label_with_episode_number(obs["pixel"], ep=ep, timestep=time_step, running_rewards=running_rewards)
            frames.append(frame)

            action, _state = model.predict(obs)
            obs, rewards, done, _info = env.step(action)
            running_rewards += rewards
            time_step += 1    
        print(f"Episode:{ep} | Total reward = {running_rewards}")
        imageio.mimwrite(f'PPO_MiniHack_Quest_Hard_ep{ep}.gif', frames, fps=30)    

if __name__ == "__main__":

    model = PPO.load(f"PPO_Trained_Model", env=env)
    render_model(model, env, 5)
    
    env.close()