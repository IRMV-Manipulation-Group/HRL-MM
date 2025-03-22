import numpy as np

import torch
from torch import nn as nn
import pytorch_util as ptu
from mlp_networks import Mlp, CNN_feature

from torch.distributions import Distribution 
from torch.distributions import Independent
from torch.distributions import Normal as TorchNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# noinspection PyMethodOverriding
class TanhGaussianPolicy(Mlp):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )

        last_hidden_size = obs_dim # if no hidden_size, dirctly connect obs_dim to action_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
        self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
    
    def get_action(self, obs_np, ):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs_np, ):
        dist = self._get_dist_from_np(obs_np)
        actions = dist.sample()
        return ptu.get_numpy(actions)
    
    def get_evaluate_action(self, obs_np):

        if torch.is_tensor(obs_np):
            h = obs_np
        else:
            h = ptu.from_numpy(obs_np)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = torch.tanh(mean)
        return mean
        # return ptu.get_numpy(mean)

    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(ptu.from_numpy(x) for x in args)
        torch_kwargs = {k: ptu.from_numpy(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist
    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        return TanhNormal(mean, std)



# noinspection PyMethodOverriding
class CNN_GaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            img_channel,
            img_width,
            img_height,
            image_feature_dim,
            hidden_width,
    ):
        super().__init__()
        self.feature_extractor = CNN_feature(
            state_dim,
            action_dim,
            img_channel,
            img_width,
            img_height,
            image_feature_dim,
            hidden_width,
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_width = hidden_width
        self.image_feature_dim = image_feature_dim

        # actor
        self.MLP = nn.Sequential(
                        nn.Linear(self.image_feature_dim + self.state_dim, self.hidden_width),
                        nn.ReLU(),
                        nn.Linear(self.hidden_width, self.hidden_width),
                        nn.ReLU(),
                        )
        
        self.last_fc = nn.Sequential(
                        nn.Linear(self.hidden_width, self.action_dim),
                        nn.Tanh()
                        )   
        self.last_fc_log_std = nn.Linear(self.hidden_width, self.action_dim)

    
    def get_action(self, s, Img1, Img2):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(ptu.device)
        Img1 = torch.unsqueeze(torch.tensor(Img1, dtype=torch.float), 0).to(ptu.device)
        Img2 = torch.unsqueeze(torch.tensor(Img2, dtype=torch.float), 0).to(ptu.device)
        dist = self.forward(s, Img1, Img2)
        action = dist.sample()[0, :]

        return ptu.get_numpy(action)
    
    def get_evaluate_action(self, s, Img1, Img2):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(ptu.device)
        Img1 = torch.unsqueeze(torch.tensor(Img1, dtype=torch.float), 0).to(ptu.device)
        Img2 = torch.unsqueeze(torch.tensor(Img2, dtype=torch.float), 0).to(ptu.device)

        Img = torch.cat((Img1, Img2), dim=2)
        h = self.feature_extractor(Img) 
        h = torch.cat((h, s), 1)
        h = self.MLP(h)
        mean = self.last_fc(h)
        # return ptu.get_numpy(mean)
        return mean


    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(ptu.from_numpy(x) for x in args)
        torch_kwargs = {k: ptu.from_numpy(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist
    

    def forward(self, s, Img1, Img2):
        Img = torch.cat((Img1, Img2), dim=2)
        h = self.feature_extractor(Img) 
        h = torch.cat((h, s), 1)
        h = self.MLP(h)
        mean = self.last_fc(h)
        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        return TanhNormal(mean, std)

class CNN_Critic(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            img_channel,
            img_width,
            img_height,
            image_feature_dim,
            hidden_width,
    ):
        super().__init__()
        self.feature_extractor = CNN_feature(
            state_dim,
            action_dim,
            img_channel,
            img_width,
            img_height,
            image_feature_dim,
            hidden_width,
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_width = hidden_width
        self.image_feature_dim = image_feature_dim

        # actor
        self.MLP = nn.Sequential(
                                nn.Linear(self.image_feature_dim + self.state_dim+self.action_dim, self.hidden_width),
                                nn.ReLU(),
                                nn.Linear(self.hidden_width, self.hidden_width),
                                nn.ReLU(),
                                nn.Linear(self.hidden_width, 1),
                                )
        
    def forward(self, s, Img1, Img2, a):
        Img = torch.cat((Img1, Img2), dim=2)
        h = self.feature_extractor(Img) 
        h = torch.cat((h, s, a), 1)
        mean = self.MLP(h)
        return mean



class CNN_GaussianPolicy1(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            img_channel,
            img_width,
            img_height,
            image_feature_dim,
            hidden_width,
    ):
        super().__init__()
        self.feature_extractor = CNN_feature(
            state_dim,
            action_dim,
            img_channel,
            img_width,
            img_height,
            image_feature_dim,
            hidden_width,
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_width = hidden_width
        self.image_feature_dim = image_feature_dim

        # actor
        self.MLP = nn.Sequential(
                        nn.Linear(self.image_feature_dim + self.state_dim, self.hidden_width),
                        nn.ReLU(),
                        nn.Linear(self.hidden_width, self.hidden_width),
                        nn.ReLU(),
                        )
        
        self.last_fc = nn.Sequential(
                        nn.Linear(self.hidden_width, self.action_dim),
                        nn.Tanh()
                        )   
        self.last_fc_log_std = nn.Linear(self.hidden_width, self.action_dim)

    
    def get_action(self, s, Img1):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(ptu.device)
        Img1 = torch.unsqueeze(torch.tensor(Img1, dtype=torch.float), 0).to(ptu.device)
        dist = self.forward(s, Img1)
        action = dist.sample()[0, :]

        return ptu.get_numpy(action)
    
    def get_evaluate_action(self, s, Img1):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(ptu.device)
        Img1 = torch.unsqueeze(torch.tensor(Img1, dtype=torch.float), 0).to(ptu.device)

        Img = Img1
        h = self.feature_extractor(Img) 
        h = torch.cat((h, s), 1)
        h = self.MLP(h)
        mean = self.last_fc(h)
        # return ptu.get_numpy(mean)
        return mean


    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(ptu.from_numpy(x) for x in args)
        torch_kwargs = {k: ptu.from_numpy(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist
    

    def forward(self, s, Img1):
        Img = Img1
        h = self.feature_extractor(Img) 
        h = torch.cat((h, s), 1)
        h = self.MLP(h)
        mean = self.last_fc(h)
        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        return TanhNormal(mean, std)

class CNN_Critic1(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            img_channel,
            img_width,
            img_height,
            image_feature_dim,
            hidden_width,
    ):
        super().__init__()
        self.feature_extractor = CNN_feature(
            state_dim,
            action_dim,
            img_channel,
            img_width,
            img_height,
            image_feature_dim,
            hidden_width,
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_width = hidden_width
        self.image_feature_dim = image_feature_dim

        # actor
        self.MLP = nn.Sequential(
                                nn.Linear(self.image_feature_dim + self.state_dim+self.action_dim, self.hidden_width),
                                nn.ReLU(),
                                nn.Linear(self.hidden_width, self.hidden_width),
                                nn.ReLU(),
                                nn.Linear(self.hidden_width, 1),
                                )
        
    def forward(self, s, Img1, a):
        Img = Img1
        h = self.feature_extractor(Img) 
        h = torch.cat((h, s, a), 1)
        mean = self.MLP(h)
        return mean
    

class MultivariateDiagonalNormal(object):
    # from torch.distributions import constraints
    # arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}

    # loc -> mu, scale -> sigma
    def __init__(self, loc, scale_diag, reinterpreted_batch_ndims=1):
        # independent is used to change the shape of the result of log_prob().
        # For example to create a diagonal Normal distribution with the same shape 
        # as a Multivariate Normal distribution (so they are interchangeable)
        # 生成独立的维度和action一样的分布
        dist = Independent(TorchNormal(loc, scale_diag),
                           reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        self.distribution = dist
    
    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def sample(self, sample_size=torch.Size()):
        return self.distribution.sample(sample_shape=sample_size)



class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        
        self.normal_mean = normal_mean
        self.normal_std = normal_std

        self.normal = MultivariateDiagonalNormal(normal_mean, normal_std)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value):
        """
        Adapted from
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73

        This formula is mathematically equivalent to log(1 - tanh(x)^2).

        Derivation:

        log(1 - tanh(x)^2)
         = log(sech(x)^2)
         = 2 * log(sech(x))
         = 2 * log(2e^-x / (e^-2x + 1))
         = 2 * (log(2) - x - log(e^-2x + 1))
         = 2 * (log(2) - x - softplus(-2x))

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        log_prob = self.normal.log_prob(pre_tanh_value)
        correction = - 2. * (
            ptu.from_numpy(np.log([2.]))
            - pre_tanh_value
            - torch.nn.functional.softplus(-2. * pre_tanh_value)
        ).sum(dim=1)

        return log_prob + correction


    def rsample_with_pretanh(self):
        z = (
                self.normal_mean +
                self.normal_std *
                MultivariateDiagonalNormal(
                    ptu.zeros(self.normal_mean.size()),
                    ptu.ones(self.normal_std.size())
                ).sample()
        )
        return torch.tanh(z), z

    def sample(self):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value.detach()

    def rsample_and_logprob(self):
        # value is the sample action after tanh
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, pre_tanh_value, log_p
class Linear_Critic(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            perception_dim,
            feature_dim,
            feature_hidden_width,
            hidden_width
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.perception_dim = perception_dim
        self.feature_dim = feature_dim
        self.feature_hidden_width = feature_hidden_width
        self.hidden_width = hidden_width

        self.feature_extractor = nn.Sequential(
                nn.Linear(self.perception_dim, self.feature_hidden_width),
                nn.ReLU(),
                nn.Linear(self.feature_hidden_width, self.feature_hidden_width),
                nn.ReLU(),
                nn.Linear(self.feature_hidden_width, self.feature_dim),
                nn.ReLU(),
                )

        self.MLP = nn.Sequential(
                                nn.Linear(self.feature_dim + self.state_dim+self.action_dim, self.hidden_width),
                                nn.ReLU(),
                                nn.Linear(self.hidden_width, self.hidden_width),
                                nn.ReLU(),
                                nn.Linear(self.hidden_width, 1),
                                )
        
    def forward(self, s, sp, a):
        h = self.feature_extractor(sp) 
        h = torch.cat((h, s, a), 1)
        mean = self.MLP(h)
        return mean

class Linear_TanhGaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            perception_dim,
            feature_dim,
            feature_hidden_width,
            hidden_width
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.perception_dim = perception_dim
        self.feature_dim = feature_dim
        self.feature_hidden_width = feature_hidden_width
        self.hidden_width = hidden_width
        
        self.feature_extractor = nn.Sequential(
                nn.Linear(self.perception_dim, self.feature_hidden_width),
                nn.ReLU(),
                nn.Linear(self.feature_hidden_width, self.feature_hidden_width),
                nn.ReLU(),
                nn.Linear(self.feature_hidden_width, self.feature_dim),
                nn.ReLU(),
                )

        self.MLP = nn.Sequential(
                        nn.Linear(self.feature_dim + self.state_dim, self.hidden_width),
                        nn.ReLU(),
                        nn.Linear(self.hidden_width, self.hidden_width),
                        nn.ReLU(),
                        )
        
        self.last_fc = nn.Sequential(
                        nn.Linear(self.hidden_width, self.action_dim),
                        nn.Tanh()
                        )   
        self.last_fc_log_std = nn.Linear(self.hidden_width, self.action_dim)

    
    def get_action(self, s, sp):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(ptu.device)
        sp = torch.unsqueeze(torch.tensor(sp, dtype=torch.float), 0).to(ptu.device)
        dist = self.forward(s, sp)
        action = dist.sample()[0, :]
        return ptu.get_numpy(action), {}
    
    def get_evaluate_action(self, s, sp):
        if torch.is_tensor(s) == False:
            s = ptu.from_numpy(s)
        if torch.is_tensor(sp) == False:
            sp = ptu.from_numpy(sp) 

        h = self.feature_extractor(sp) 

        if len(h.size()) == 2:
            h = torch.cat((h, s), 1)
        else:
            h = torch.cat((h, s))

        h = self.MLP(h)

        mean = self.last_fc(h)
        # return ptu.get_numpy(mean)
        return mean



    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(ptu.from_numpy(x) for x in args)
        torch_kwargs = {k: ptu.from_numpy(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist
    

    def forward(self, s, sp):
        h = self.feature_extractor(sp) 
        h = torch.cat((h, s), 1)
        h = self.MLP(h)
        mean = self.last_fc(h)
        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        return TanhNormal(mean, std)
