import helper

from functools import reduce

import torch
import numpy as np
import gym
from tqdm import tqdm

from memory_collector import MemoryCollector
from ppo2 import PPO2

def build_train_fn(train_model, optimizer, device):
    def loss_fn(reward, value_f, neg_log_prob, entropy, advantages, old_value_f, old_neg_log_prob, clip_range=0.2, ent_coef=0.01, vf_coef=1):
        """
        compute loss
        :param reward: total reward obtained
        :param value_f: estimated value function
        :param neg_log_prob: negative log-likelihood of each action
        :param entropy: entropy of the action distribution
        :param advantages: estimated advantage of each action
        :param old_value_f: estimated value function using old parameters
        :param old_neg_log_prob: negative log-likelihood of each action using old parametres
        :param clip_range: policy clip value
        :param ent_coef: entropy discount coefficient
        :param vf_coef: value discount coefficient
        :return: total loss, policy loss, value loss, entropy, approximated KL-div between new and old action distribution
        """

        # value loss
        value_loss = .5 * torch.mean((value_f - reward)**2)

        # approximated KL-divergence (actually of no use in PPO2)
        approx_kl = .5 * torch.mean((neg_log_prob - old_neg_log_prob)**2)

        # PPO2 loss
        ratio = torch.exp(old_neg_log_prob - neg_log_prob)
        normal_pg_loss = -advantages * ratio
        cliped_pg_loss = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        pg_loss = torch.mean(torch.max(normal_pg_loss, cliped_pg_loss))
        
        # entropy loss
        entropy_mean = entropy.mean()
        loss = pg_loss - (entropy_mean * ent_coef) + (value_loss * vf_coef)


        return loss, pg_loss, value_loss, entropy_mean, approx_kl

    def train_step_fn(obs, returns, dones, old_actions, old_values, old_neg_log_prbs):
        obs = torch.tensor(obs).float().to(device)
        returns = torch.tensor(returns).float().to(device)
        old_values = torch.tensor(old_values).float().to(device)
        old_neg_log_prbs = torch.tensor(old_neg_log_prbs).float().to(device)
        old_actions = torch.tensor(old_actions).to(device)

        with torch.set_grad_enabled(False):
            # Normalize the advantages
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        train_model.train()
        with torch.set_grad_enabled(True):
            train_model.zero_grad()
            value_f, actions, neg_log_probs, entropy = train_model(obs, action=old_actions)
            loss, pg_loss, value_loss, entropy_mean, approx_kl = loss_fn(returns, value_f, neg_log_probs, entropy, advantages,
                                                                               old_values, old_neg_log_prbs)
            loss.backward()
            optimizer.step()

        return list(map(lambda x: x.detach().item(), [loss, pg_loss, value_loss, entropy_mean, approx_kl]))

    return train_step_fn

if __name__ == '__main__':
    # set device
    device = torch.device("cpu")

    # create environment
    env = gym.make('LunarLander-v2', render_mode="rgb_array")

    # compute model hyper-parameter
    obs_size = reduce((lambda x, y: x * y), env.observation_space.shape)
    action_space = env.action_space.n

    # define model
    model = PPO2(input_dim=obs_size, hidden_dim=32, action_space=action_space, dropout=0.0)
    model.to(device)

    # define optimizer
    optm = torch.optim.Adam(params=model.parameters(), lr=3e-4, eps=1e-8)

    # setup training function
    train_fn = build_train_fn(model, optm, device)

    # create memory collector for different episode. Used for batch training
    memory_collector = MemoryCollector(env=env, model=model, device=device)

    # training parameters
    n_updates = 10000

    for update in tqdm(range(1, n_updates+1)):
        # collect episodes for training
        obs, returns, dones, actions, values, neg_log_prb, final_returns, rewards_mean = memory_collector.run(10, max_step=1000, splitting=64)

        # log the loss
        avg_loss = []
        avg_kl = []
        avg_entropy = []

        # train model (batch segmented from the collected episodes)
        for slices in zip(obs, returns, dones, actions, values, neg_log_prb):
            loss, pg_loss, value_loss, entropy, approx_kl = train_fn(*slices)

            avg_loss.append(loss)
            avg_kl.append(approx_kl)
            avg_entropy.append(entropy)

        # visualize the training
        print('-'*30)
        print("MISC n_updates", update)
        print("MISC loss", np.mean(avg_loss))
        print("MISC approx_kl", np.mean(avg_kl))
        print("MISC entropy", np.mean(avg_entropy))
        print('-'*10)
        print("ENVS return_mean", np.mean(final_returns))
        print("ENVS rewards_mean", np.mean(rewards_mean))
        print('-'*30)

        # save model checkpoint
        helper.save_checkpoint({
            'update': update,
            'state_dict': model.state_dict(),
            'optimizer': optm.state_dict()
        },
            path='./model',
            filename='train_net.cptk',
            version='0.0.1'
        )
        
        # render the environment
        memory_collector.run(1, max_step=1000, render=True)

    env.close()