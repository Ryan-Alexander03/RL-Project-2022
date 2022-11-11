from make_env import make_env

import os
from datetime import datetime
from stable_baselines3 import PPO

"""
Training
"""
# Creat folders to save the training long and saving model
dt_string = datetime.now().strftime("%d-%b")
log_dir = f"{dt_string}/log/"
os.makedirs(log_dir, exist_ok=True)
model_dir = f"{dt_string}/saved_models/"
os.makedirs(model_dir, exist_ok=True)

# Logging the training output
log_name = "PPO_batch_size=64_n_steps=512_n_epochs = 16"

# Save the model per 100,000 learning steps
L_steps = 100000

# initialse the model with tuned parameters
env = make_env()
model = PPO("MultiInputPolicy", env=env, verbose=1, tensorboard_log = log_dir, batch_size=64, n_steps=512, clip_range=0.2, n_epochs = 16)

# start training cycle
if __name__ == "__main__":
        for i in range(200): # train and SAVE per learning steps
                model.learn(total_timesteps = L_steps, reset_num_timesteps = False, tb_log_name = log_name)
                print("SAVING MODEL.")
                model.save(f"{model_dir}/{log_name}/PPO_Trained_Model")
                print("MODEL SAVED.")
                
        env.close()
