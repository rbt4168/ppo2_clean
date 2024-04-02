import argparse
import helper

from datetime import datetime
from functools import reduce

import torch
import numpy as np

from tqdm import tqdm
import gym

from memory_collector import MemoryCollector
from model.ppo2 import ppo2
# from env.env import StockTradingEnv

EXP_NAME = "exp-ppo-{}".format(datetime.now().strftime("%H:%M:%S"))

def __pars_args__():
    parser = argparse.ArgumentParser(description='PPO')

    parser.add_argument('-gam', '--gamma', type=float, default=0.95, help='discounting factor')
    parser.add_argument('-lam', '--lam', type=float, default=0.99, help='discounting factor')
    parser.add_argument('-m_grad', '--max_grad_norm', type=float, default=0.5, help='max norm of the gradients')
    parser.add_argument('-ent_coef', '--ent_coef', type=float, default=0.,
                        help='policy entropy coefficient in the optimization objective')
    parser.add_argument('-vf_coef', '--vf_coef', type=float, default=0.5,
                        help='value function loss coefficient in the optimization objective')
    parser.add_argument("-tot_t", "--total_timesteps", type=int, default=1000000,
                        help="number of timesteps (i.e. number of actions taken in the environment)")

    parser.add_argument('-n_train_ep', '--number_train_epoch', type=int, default=1,
                        help='number of training epochs per update')
    parser.add_argument('-clp_rng', '--clip_range', type=float, default=0.2,
                        help='clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the ' +
                             'training and 0 is the end of the training')


    parser.add_argument('--lstmseed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4, help='learning rate (default: 0.001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='batch size used during learning')
    parser.add_argument('-mini_bs', '--mini_batchs', type=int, default=1, help='number of training minibatches per update. For recurrent policies, should be smaller or equal than number of environments run in parallel.')
    parser.add_argument('-n_step', '--n_step', type=int, default=8192, help='number of steps of the vectorized environment per update')

    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')

    parser.add_argument('-m_path', '--model_path', default='./trained_model', help='Path to save the model')
    parser.add_argument('-v_path', '--monitor_path', default='./monitor', help='Path to save monitor of agent')
    parser.add_argument('-v', '--version', default='0', help='Path to save monitor of agent')

    parser.add_argument('-save_every', '--save_every', type=int, default=1,
                        help='number of timesteps between saving events')

    parser.add_argument('-log_every', '--log_every', type=int, default=1,
                        help='number of timesteps between logs events')


    parser.add_argument("--max_steps", type=int, default=100, help="Max step for an episode")
    parser.add_argument("--state_size", type=list, default=[56, 84], help="Frame size")
    parser.add_argument("-uc", "--use_cuda", type=bool, default=True, help="Use cuda")

    return parser.parse_args()

def build_train_fn(args, train_model, optimizer, device):
    def loss_fn(reward, value_f, neg_log_prob, entropy, advantages, old_value_f, old_neg_log_prob, clip_range, ent_coef, vf_coef):
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
        value_f_clip = old_value_f + torch.clamp(value_f - old_value_f, min=-clip_range, max=clip_range)

        normal_value_loss_square = (value_f - reward)**2
        cliped_value_loss_square = (value_f_clip - reward)**2
        # value_loss = F.smooth_l1_loss(value_f, reward)
        value_loss = .5 * torch.mean(torch.max(normal_value_loss_square, cliped_value_loss_square)) # 1/2 * max((value_f - reward)^2, (value_f_clip - reward)^2)

        ratio = torch.exp(old_neg_log_prob - neg_log_prob)

        normal_pg_loss = -advantages * ratio
        cliped_pg_loss = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        pg_loss = torch.mean(torch.max(normal_pg_loss, cliped_pg_loss))
        
        # clip_frac = (torch.abs(ratio - 1.0) > clip_range).float().mean()
        entropy_mean = entropy.mean()
        loss = pg_loss - (entropy_mean * ent_coef) + (value_loss * vf_coef)

        approx_kl = .5 * torch.mean((neg_log_prob - old_neg_log_prob)**2) # MSE between new and old action distribution

        return loss, pg_loss, value_loss, entropy_mean, approx_kl

    def train_step_fn(obs, returns, dones, old_actions, old_values, old_neg_log_prbs):
        assert old_neg_log_prbs.min() > 0

        obs = torch.tensor(obs).float().to(device)
        returns = torch.tensor(returns).float().to(device)
        old_values = torch.tensor(old_values).float().to(device)
        old_neg_log_prbs = torch.tensor(old_neg_log_prbs).float().to(device)
        old_actions = torch.tensor(old_actions).to(device)

        with torch.set_grad_enabled(False):
            advantages = returns - old_values
            # Normalize the advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        train_model.train()
        with torch.set_grad_enabled(True):
            train_model.zero_grad()

            value_f, actions, neg_log_probs, entropy = train_model(obs, action=old_actions)

            assert(actions.sum().item() == old_actions.sum().item())

            loss, pg_loss, value_loss, entropy_mean, approx_kl = loss_fn(returns, value_f, neg_log_probs, entropy, advantages,
                                                                               old_values, old_neg_log_prbs,
                                                                               args.clip_range, args.ent_coef, args.vf_coef)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
            optimizer.step()

        return list(map(lambda x: x.detach().item(), [loss, pg_loss, value_loss, entropy_mean, approx_kl]))

    return train_step_fn

if __name__ == '__main__':
    args = __pars_args__()

    # set device
    device = torch.device("cuda:0" if args.use_cuda else "cpu")

    # create environment
    env = gym.make('LunarLander-v2')

    # compute model hyper-parameter
    obs_size = reduce((lambda x, y: x * y), env.observation_space.shape)
    action_space = env.action_space.n

    # define model
    model = ppo2(reset_param=True, input_dim=obs_size, hidden_dim=32, action_space=action_space, dropout=0.0)
    model.to(device)

    # define optimizer
    optm = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, eps=1e-8)

    # setup training function
    train_fn = build_train_fn(args, model, optm, device)

    # create memory collector for different episode. Used for batch training
    memory_collector = MemoryCollector(env=env, model=model, device=device)

    # training parameters
    n_updates = 10000

    for update in tqdm(range(1, n_updates+1)):
        # collect episodes for training
        obs, returns, dones, actions, values, neg_log_prb, final_returns = memory_collector.run(10, max_step=1000, splitting=64)

        # normalize the returns
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # print(obs)

        # log the loss
        avg_loss = []
        avg_kl = []

        # train model (batch segmented from the collected episodes)
        # loss, pg_loss, value_loss, entropy, approx_kl = train_fn(obs, returns, dones, actions, values, neg_log_prb)
        for slices in zip(obs, returns, dones, actions, values, neg_log_prb):
            loss, pg_loss, value_loss, entropy, approx_kl = train_fn(*slices)

            avg_loss.append(loss)
            avg_kl.append(approx_kl)

        if update % args.log_every == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            print('-'*30)
            # ev = helper.explained_variance(values, returns)
            print("MISC n_updates", update)
            # print("MISC explained_variance", float(ev))
            print("MISC loss", np.mean(avg_loss))
            print("MISC approx_kl", np.mean(avg_kl))
            print('-'*10)
            print("MISC return", final_returns)
            print('-'*30)

        if update % args.save_every == 0 or update == 1:
            # save model checkpoint
            helper.save_checkpoint({
                'update': update,
                'state_dict': model.state_dict(),
                'optimizer': optm.state_dict()
            },
                path=args.model_path,
                filename='train_net.cptk',
                version=args.version
            )

    env.close()