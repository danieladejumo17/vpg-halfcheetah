import torch
import torch.nn as nn
import numpy as np
import gym

import vpg.vpg_core as core
from vpg.vpg_buffer import VPGBuffer
from vpg.train_vpg import train_vpg


# def compute_loss_pi(data, ac):
#     # obs, act, ret, adv, logp = data
#     obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

#     pi, logp = ac.pi(obs, act)
#     loss_pi = -(logp*adv).mean()

#     # Compute KL divergence and entropy
#     approx_kl = (logp_old - logp).mean().item()
#     ent = pi.entropy().mean().item()
#     pi_info = dict(kl=approx_kl, ent=ent)
    
#     return loss_pi, pi_info


# def compute_loss_v(data, ac):
#     # obs, act, ret, adv, logp = data
#     obs, ret = data["obs"], data["ret"]
#     return ((ac.v(obs) - ret)**2).mean()


# def update(data, ac, pi_optimizer, vf_optimizer, train_v_iters):
#     # Get loss and info values before update
#     loss_pi_old, pi_info_old = compute_loss_pi(data, ac)
#     loss_pi_old = loss_pi_old.item()
#     loss_v_old = compute_loss_v(data, ac).item()
    
#     # Train the policy with a single step of gradient descent
#     pi_optimizer.zero_grad()
#     loss_pi, pi_info = compute_loss_pi(data, ac)
#     loss_pi.backward()
#     pi_optimizer.step()

#     # Fit value function
#     for _  in range(train_v_iters):
#         vf_optimizer.zero_grad()
#         loss_v = compute_loss_v(data, ac)
#         loss_v.backward()
#         vf_optimizer.step()
        
#     # Log changes from update
#     kl, ent = pi_info["kl"], pi_info_old["ent"]
#     return dict(LossPi=loss_pi_old, LossV=loss_v_old, KL=kl, Entropy=ent, 
#                 DeltaLossPi=loss_pi.item() - loss_pi_old, DeltaLossV=loss_v.item() - loss_v_old)


# from torch.utils.tensorboard import SummaryWriter

# def train_vpg(env_fn, actor_critic, ac_kwargs, pi_lr, vf_lr,
#               epochs, steps_per_epoch, gamma, gae_lambda, train_v_iters,
#               max_ep_len, log_freq=10, seed=0, exp_name="vpg"):
#     # set seed
#     torch.manual_seed(seed)
#     np.random.seed(seed)
    
#     # Create the training environment
#     env = env_fn()
    
#     # Create the actor-critic
#     ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
#     param_counts = tuple(core.count_params(module) for module in [ac.pi, ac.v])
    
    
#     # Create optimizers for the policy and value function
#     pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
#     vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)
    
#     # Create the VPG Buffer
#     buffer = VPGBuffer(env.observation_space.shape, env.action_space.shape, 
#                        steps_per_epoch, gamma, gae_lambda)
    
#     # Tensorboard Writer
#     writer = SummaryWriter(f"tensorboard/{exp_name}")
    
#     # Run `epochs` number of epochs
#     obs, ep_ret, ep_len = env.reset(), 0, 0
#     for epoch in range(epochs):
#         epoch_rets, epoch_lens = [], []
#         for step in range(steps_per_epoch):
#             # get action from the policy
#             act, val, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))

#             # step through the env with the action from the policy
#             next_obs, rew, done, _ = env.step(act)
#             ep_ret += rew
#             ep_len += 1

#             # store step in the VPG buffer
#             buffer.store(obs, act, rew, val, logp)

#             # update the obs
#             obs = next_obs

#             # check for terminal or epoch end
#             timeout = (ep_len == max_ep_len)
#             terminal = (done or timeout)
#             epoch_end = (step == (steps_per_epoch - 1))

#             # if trajectory ends or epoch ends
#             if terminal or epoch_end:
#                 # Log trajectory cut-off byb epoch end
#                 if epoch_end and not terminal:
#                     pass
#                     # print(f"WARNING: Trajectory cut off by epoch end at step {ep_len}")

#                 # bootstrap value target if trajectory didn't reach terminal state
#                 if timeout or epoch_end: # change to if not done
#                     _, v, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
#                 else:
#                     v = 0

#                 # Finish a trajectory
#                 buffer.finish_path(v)

#                 # only save ep_rew and ep_len if trajectory finished
#                 if terminal:
#                     epoch_lens.append(ep_len)
#                     epoch_rets.append(ep_ret)
#                 obs, ep_len, ep_ret = env.reset(), 0, 0
            
#         # Perform VPG update
#         data = buffer.get()
#         res = update(data, ac, pi_optimizer, vf_optimizer, train_v_iters)
        
#         # Log Result
#         if (epoch % log_freq == 0) or (epoch == epochs - 1):
#             print(f"Epoch: {epoch} Mean Reward: {np.mean(epoch_rets):.2f}, Mean Length: {np.mean(epoch_lens):.1f} LossV: {res['LossV']:.3f}")
            
#         writer.add_scalar("Mean Return", np.mean(epoch_rets), global_step=epoch)
#         writer.add_scalar("Mean Length", np.mean(epoch_lens), global_step=epoch)
#         writer.add_scalar("Value Loss", res['LossV'], global_step=epoch)
    
#     return ac

if __name__ == "__main__":
    # HYPERPARAMETERS
    ENV_ID = "HalfCheetah-v2"   # "CartPole-v1"
    HIDDEN_SIZES = [64, 64]  # for both pi and v
    PI_LR = 1e-3
    V_LR = 1e-3
    GAMMA = 0.99
    GAE_LAMBDA = 0.98
    ACTIVATION_FN = nn.Tanh
    TRAIN_V_ITERS = 20
    MAX_EPISODE_LENGTH = 1000
    STEPS_PER_EPOCH = 4000
    EPOCHS = 1
    SEED = 0


    # Set SEEDS
    torch.manual_seed(SEED)
    np.random.seed(SEED)


    # Create the env
    env = gym.make(ENV_ID)

    # Create the Actor-Critic
    ac = core.MLPActorCritic(env.observation_space, env.action_space, HIDDEN_SIZES, activation=nn.Tanh)

    # Create optimizers for the actor and critic models
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=PI_LR)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=V_LR)

    # Create the VPGBuffer Object
    buf = VPGBuffer(env.observation_space.shape, env.action_space.shape, STEPS_PER_EPOCH, GAMMA, GAE_LAMBDA)

    # Count Paramaters
    var_counts = tuple(core.count_params(module) for module in [ac.pi, ac.v])
    print("Number of Parameters. PI: {} V: {}".format(*var_counts))

    import time
    train_kwargs = {"env_fn": lambda: gym.make(ENV_ID), 
                "actor_critic": core.MLPActorCritic,
                "ac_kwargs": {"hidden_sizes": HIDDEN_SIZES, "activation": ACTIVATION_FN},
                "pi_lr": PI_LR,
                "vf_lr": V_LR,
                "epochs": EPOCHS,
                "steps_per_epoch": STEPS_PER_EPOCH,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "train_v_iters": TRAIN_V_ITERS,
                "max_ep_len": MAX_EPISODE_LENGTH, 
                "log_freq": 10, 
                "seed": SEED, 
                "exp_name": f"vpg_{ENV_ID}_{time.time()}"}

    model = train_vpg(**train_kwargs)

    # SAVE THE MODEL
    torch.save(model.state_dict(), f"./logs/checkpoints/{ENV_ID}_{time.time()}.pth")

    # SEE THE AGENT PLAY
    NUM_EPISODES = 10
    done = False
    obs = env.reset()

    for _ in range(NUM_EPISODES):
        while not done:
            act = model.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)
            env.render()
        done = False
        obs = env.reset()
    env.close()
