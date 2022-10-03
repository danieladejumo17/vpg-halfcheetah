import torch
import torch.nn as nn
import numpy as np
import gym

import vpg.vpg_core as core
from vpg.vpg_buffer import VPGBuffer
from vpg.train_vpg import train_vpg


if __name__ == "__main__":
    # HYPERPARAMETERS
    ENV_ID = "HalfCheetah-v2"
    HIDDEN_SIZES = [64, 64]  # for both pi and v
    ACTIVATION_FN = nn.Tanh
    
    
    # CREATE THE ENVIRONMENT
    env = gym.make(ENV_ID)
    

    # LOAD A SAVED
    ac_kwargs = {"hidden_sizes": HIDDEN_SIZES, "activation": ACTIVATION_FN}
    model = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    model.load_state_dict(torch.load(f"./logs/checkpoints/{ENV_ID}.pth"))
    
    
    # SEE THE AGENT PLAY
    NUM_EPISODES = 10
    done = False
    obs = env.reset()

    for episode in range(NUM_EPISODES):
        while not done:
            act = model.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)
            env.render()
        done = False
        obs = env.reset()
        print(f"Completed episode {episode}")
    env.close()