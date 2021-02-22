import torch
import numpy as np
from bandit.modules.module import Module
from bandit.parameter import Parameter
from bandit.containers.container import Container
from collections import OrderedDict
from copy import deepcopy
import cvxpy

MAX_ITER = 100000


class TCGP(Module):
    """
    OLU class, which is a data structure of parameter
    :param self.w: weight of portfolio
    :param self.eta: learning rate
    :param self.alpha: alpha = eta * gamma, gamma is the transaction cost rate
    :param self.beta: the parameter for the augmentation term of the OLU algorithm
    """

    # __constants__ = ['w', 'eta']  # ?

    def __init__(self, module_id=-1, eta=20, gamma=0.1, beta=0.1, k=10, **kwargs):
        super(TCGP, self).__init__(module_id=module_id)
        self.k = k
        self.w = Parameter(torch.ones(k) * 1 / k)
        self.eta = torch.tensor(eta)
        self.alpha = torch.tensor(gamma * eta)
        self.beta = torch.tensor(beta)
        self.register_decide_hooks(name_list=['w'])

    def init(self, k):
        """
        For re-init when changing k
        :param k: portfolio size
        :return: None
        """
        self.w = Parameter(torch.ones(k) * 1 / k)
        self.k = k
        self.register_decide_hooks(name_list=['w'])

    def update(self, arm_reward, last_weight=None, last_weight_hat=None, prob=1., *args, **kwargs):
        # prob是reward的estimator概率
        # w_t是上一次的权重
        if last_weight is None:
            w_t = deepcopy(self.w)
        else:
            w_t = deepcopy(self.w)
            w_t_hat = deepcopy(last_weight_hat)
            # print("pre:", w_t, w_t_hat)

        z = torch.zeros(self.k)
        u = torch.zeros(self.k)
        shrinkage = lambda x: torch.max(torch.zeros(1), x - self.alpha / self.beta) - torch.max(torch.zeros(1),
                                                                                                -x - self.alpha / self.beta)

        for i in range(MAX_ITER):
            old_z = deepcopy(z)
            v = self.eta * prob * arm_reward / ((self.beta + 1) * torch.dot(w_t, arm_reward)) + (torch.tensor(1.) / (
                self.beta + 1)) * w_t + (self.beta / (self.beta + 1)) * w_t_hat + self.beta * (z - u) / (self.beta + 1)
            self.w = self.__project_to_simplex(v)  # projection to the simplex

            z = shrinkage(self.w - w_t_hat + u)
            r = self.w - w_t_hat - z
            u = u + r
            s = self.beta * (z - old_z)
            if torch.norm(r) < 1e-3 and torch.norm(s) < 1e-3:
                # print("out:", self.w)
                # print(torch.abs(self.w-w_t).sum())
                break

    @staticmethod
    def __project_to_simplex(v):
        n = v.shape[0]
        v = v.numpy()
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        # get the number of > 0 components of the optimal solution
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssv[rho] - 1) / (rho + 1.0)
        # compute the projection by thresholding v using theta
        w = (v - theta).clip(min=0)
        w = torch.tensor(w)
        return w


if __name__ == '__main__':
    from bandit.containers.portfolio import Portfolio
    from utils.data import BanditData

    print("hello world")
    olu = Portfolio(container_id=1, module={'file': "olu", 'name': 'OLU'}, k=2)
    w = olu(BanditData(arm_reward={1: torch.tensor(1.), 2: torch.tensor(0.9)}))
    print(w)
