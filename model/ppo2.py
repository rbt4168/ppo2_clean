import torch
import torch.nn as nn
import torch.distributions as D


__all__ = ['PPO2', 'ppo2']


class PPO2(nn.Module):
    def __init__(self, input_dim, hidden_dim,  action_space, dropout):
        """
        ppo2 model
        :param input_dim: observation dimension
        :param hidden_dim: hidden state dimension
        :param action_space: action space
        :param dropout: dropout probability
        """
        super(PPO2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.dropout = dropout

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Tanh()
        )

        self.policy_head = nn.Linear(self.hidden_dim, self.action_space)
        self.value_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, action=None, distribution=D.categorical.Categorical):
        """
        compute latent state representation.
        such state representation is then used to execute the policy and estimate the value function
        :param x: input observation
        :param action: if None, execute the new policy. Otherwise, compute the negative log-likelihood of the new policy
        :param distribution: distribution type for the given action. Have to be a torch.distributions
        :return:
        """

        if not issubclass(distribution, torch.distributions.distribution.Distribution):
            raise NotImplementedError("distribution type have to be a valid torch.distribution class (logits of each action are used instead of probabilities).")

        latent_state = self.network(x)
        action_logit = self.policy_head(latent_state)   # might change during training. thus we recompute the neg_log_prob
        action_dist = distribution(logits=action_logit)

        if action is None:
            action = action_dist.sample()
            neg_log_prob = action_dist.log_prob(action) * -1.
            entropy = action_dist.entropy()
        else:
            neg_log_prob = action_dist.log_prob(action) * -1.
            entropy = action_dist.entropy()

        value_f = self.value_head(latent_state)

        return value_f, action, neg_log_prob, entropy

    def reset_parameters(self):
        """
        randomly reset the model's parameters
        :return:
        """
        for p in self.parameters():
            if len(p.data.shape) == 2:
                # hidden layer
                nn.init.orthogonal_(p, gain=2**0.5)
            elif len(p.data.shape) == 1:
                # bias
                nn.init.constant_(p, 0.0)


def ppo2(reset_param=True, **kwargs):
    """
    ppo2 model b ased on the implementation proposed at https://github.com/openai/baselines/tree/master/baselines/ppo2.
    Paper: https://arxiv.org/abs/1707.06347
    :param reset_param: it True, randomly reset the initial parameters according using a (semi) orthogonal matrix.
    """
    model = PPO2(**kwargs)
    if reset_param:
        model.reset_parameters()

    return model