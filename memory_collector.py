import numpy as np
import torch

import matplotlib.pyplot as plt
from IPython import display

class MemoryCollector(object):
    """
    We use this object to make a episode of experiences
    __init__:
    - Initialize the runner
    run():
    - Make a episode of experiences
    """
    def __init__(self, env, model, device, gamma=0.99, lam=0.99):
        self.env = env
        self.model = model

        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam

        # Discount rate
        self.gamma = gamma
        self.device = device

    def show_state(self, step=0, info=""):
        plt.clf()
        plt.imshow(self.env.render())
        plt.title("%s | Step: %d %s" % (self.env.spec.id, step, info))
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.01)

    def eval_fn(self, obs):
        """
        evaluation function. Choose an action based on the current policy
        :param obs: environment observation
        :return:
        """
        self.model.eval()
        with torch.set_grad_enabled(False):
            obs = torch.tensor(obs).float().to(self.device)
            value_f, action, neg_log_prob, entropy = self.model(obs)
        return value_f, action, neg_log_prob

    def run(self, n_episodes, max_step=None, splitting=1, render=False):
        """
        Collect experiences and return multi episodes of experiences

        :param n_episodes: number of episodes to collect
        :param max_step: maximum number of steps to collect

        :return: multi episodes of experiences
        """
        mb_obs, mb_rewards, mb_actions, mb_values, mb_done, mb_neg_log_prob = [],[],[],[],[],[]

        mb_returns = []
        mb_rewards_mean = []

        for _ in range(n_episodes):
            s_obs, s_rewards, s_done, s_actions, s_values, s_neg_log_prob, s_reward = self._run(max_step, render=render)
            
            mb_returns.append(s_rewards[-1])
            mb_rewards_mean.append(np.sum(s_reward))
            # Extend the lists with the chunks
            mb_obs.extend(s_obs)
            mb_rewards.extend(s_rewards)
            mb_done.extend(s_done)
            mb_actions.extend(s_actions)
            mb_values.extend(s_values)
            mb_neg_log_prob.extend(s_neg_log_prob)

        splitting_count = len(mb_obs) // splitting

        mb_obs = np.array_split(np.asarray(mb_obs), splitting_count)
        mb_rewards = np.array_split(np.asarray(mb_rewards), splitting_count)
        mb_done = np.array_split(np.asarray(mb_done), splitting_count)
        mb_actions = np.array_split(np.asarray(mb_actions), splitting_count)
        mb_values = np.array_split(np.asarray(mb_values), splitting_count)
        mb_neg_log_prob = np.array_split(np.asarray(mb_neg_log_prob), splitting_count)

        return mb_obs, mb_rewards, mb_done, mb_actions, mb_values, mb_neg_log_prob, mb_returns, mb_rewards_mean
    
    def _run(self, max_step=None, render=False):
        obs, _ = self.env.reset()
        done = False

        if render:
            self.show_state()

        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_done, mb_neg_log_prob = [],[],[],[],[],[]

        # For n in range number of steps
        actual_step = 0
        while not done:
            obs = np.expand_dims(obs, 0)
            mb_obs.append(obs.copy())

            # compute step
            value_f, action, neg_log_prob = self.eval_fn(obs)

            # values for training
            action = action.cpu().item()
            neg_log_prob = neg_log_prob.cpu().item()
            value_f = value_f.cpu().item()

            mb_actions.append(action)
            mb_neg_log_prob.append(neg_log_prob)
            mb_values.append(value_f)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            obs, rewards, done, _, info = self.env.step(action)

            if render:
                self.show_state(actual_step, info)

            mb_rewards.append(rewards)
            mb_done.append(done)

            actual_step += 1

            if max_step is not None and actual_step >= max_step:
                break

        # batch of steps to batch of rollouts
        mb_obs = np.concatenate(mb_obs, 0).astype(np.float32)

        mb_actions = np.asarray(mb_actions)
        mb_neg_log_prob = np.asarray(mb_neg_log_prob, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)

        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_done = np.asarray(mb_done, dtype=np.bool)

        # get value function for last action
        obs = np.expand_dims(obs, 0)
        last_values, _, _ = self.eval_fn(obs)
        last_values = last_values.cpu().item()

        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(actual_step)):
            if t == actual_step - 1:
                next_non_terminal = 1.0 - done
                next_values = last_values
            else:
                next_non_terminal = 1.0 - mb_done[t+1]
                next_values = mb_values[t+1]

            delta = mb_rewards[t] + self.gamma * next_values * next_non_terminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * next_non_terminal * lastgaelam

        # compute value functions
        mb_returns = mb_advs + mb_values

        return mb_obs, mb_returns, mb_done, mb_actions, mb_values, mb_neg_log_prob, mb_rewards