import torch
from bandit.modules.module import Module
from bandit.functional import draw


class Exp4(Module):
    """
    Exp4 class, which is a data structure of parameter
    :param self.w: weight of portfolio
    :param self.gamma: tradeoff parameter
    """
    __constants__ = ['w', 'gamma', 'm', 'k', 'p']

    def __init__(self, module_id=-1, total_round=100, arms=1, g_max=1., m=1, **kwargs):
        super(Exp4, self).__init__(module_id=module_id)
        self.w = torch.ones(m)
        self.m = m

        def exp4_gamma(k, g_t, t, m):
            gamma = torch.sqrt(k * torch.log(m) / ((torch.exp(torch.tensor(1.)) - 1.) * t * g_t))
            if gamma > 1:
                return torch.tensor(1.)
            else:
                return gamma

        self.gamma = exp4_gamma(torch.tensor(arms * 1.), torch.tensor(g_max * 1.), torch.tensor(total_round * 1.), m * 1.)
        self.k = torch.tensor(arms)
        self.p = {}

    def update(self, result, arm_reward, zetas):
        if self.p.get(result) is None:
            p = self.p["remains"]
        else:
            p = self.p[result]
        x_j = arm_reward / p
        idx = 0
        for zeta in zetas:
            if zeta.get(result) is None:
                p_zeta = zeta["remains"]
            else:
                p_zeta = zeta[result]
            y = p_zeta * x_j
            self.w[idx] *= torch.exp(y * self.gamma / self.k)
            idx += 1

    def decide(self, zetas, combinarial_bandit, arm_set=None, **kwargs):
        if combinarial_bandit:
            self.p = self.distr_exp4(k=self.k, zetas=zetas)
            recommend_arm = draw(self.p, k=self.k)
        else:
            self.p = self.distr_exp4(arm_set=arm_set, zetas=zetas)
            recommend_arm = draw(self.p, arm_set=arm_set)
        if recommend_arm in self.p.keys():
            prob = self.p[recommend_arm]
        else:
            prob = self.p["remains"]
        return recommend_arm, prob

    def distr_exp4(self, zetas, arm_set=None, k=None):
        """
        when arm_set is not None: density
        when k is not None: sparsity (for combinarial bandit)
        Input:  weights = array(1-by-M)
                M: num of experts

                zetas = list(N-by-dict)
                zetas[i][idx_combination] = value
                dict['remains'] = remaining prob value

        Output: res = dict
                res[idx_combination] = value
                res['remains'] = remaining prob value (sparsity)
        """
        res = {}

        if arm_set is not None:
            the_sum = self.w.sum()
            w = self.w / the_sum
            for arm in arm_set:
                prob = 0.
                for i in range(self.m):
                    if arm in zetas[i].keys():
                        prob += w[i] * zetas[i][arm]
                res[arm] = self.__exponential_weighting(prob, self.gamma, len(arm_set))
        else:
            the_sum = self.w.sum()
            w = self.w / the_sum

            not_remain_idx = set()
            for i in range(self.m):
                not_remain_idx = not_remain_idx.union(zetas[i])

            for idx in not_remain_idx:
                prob = 0.
                for i in range(self.m):
                    if idx in zetas[i]:
                        prob += zetas[i][idx] * w[i]
                    else:
                        prob += zetas[i]['remains'] * w[i]
                res[idx] = self.__exponential_weighting(prob, self.gamma, k)

            if len(not_remain_idx) < k:
                prob = 0.
                for i in range(self.m):
                    prob += zetas[i]['remains'] * w[i]
                res['remains'] = self.__exponential_weighting(prob, self.gamma, k)
        return res

    @staticmethod
    def __exponential_weighting(prob, gamma, k):
        p = (torch.tensor(1.0) - gamma) * prob + gamma / k
        return p


class Exp4LossBased(Exp4):
    __constants__ = ['w', 'gamma', 'k', 'p']

    def __init__(self, module_id=-1, total_round=100, arms=1, g_max=1, m=1, **kwargs):
        super(Exp4LossBased, self).__init__(module_id=module_id, total_round=total_round, arms=arms, g_max=g_max, m=m, **kwargs)
        self.g_max = g_max

        def exp4lossbased_eta(m, k, t):
            return torch.sqrt(2. * torch.log(m) / (t * k))

        self.eta = exp4lossbased_eta(m * 1., arms, total_round)

    def update(self, result, arm_reward, zetas):
        if self.p.get(result) is None:
            p = self.p["remains"]
        else:
            p = self.p[result]
        x_j = 1 - (1 - arm_reward/self.g_max) / p
        idx = 0
        for zeta in zetas:
            if zeta.get(result) is None:
                p_zeta = zeta["remains"]
            else:
                p_zeta = zeta[result]
            y = p_zeta * x_j
            self.w[idx] *= torch.exp(y * self.eta)
            idx += 1

    @staticmethod
    def __exponential_weighting(prob, gamma, k):
        p = prob
        return p


class LazyExp4(Exp4LossBased):
    __constants__ = ['w', 'gamma', 'k', 'p']

    def __init__(self, module_id=-1, total_round=100, arms=1, g_max=1, m=1, stage=100,  **kwargs):
        super(LazyExp4, self).__init__(module_id=module_id, total_round=total_round, arms=arms, g_max=g_max, m=m, **kwargs)
        self.recommend_arm = None
        self.lazy_time = total_round / stage
        # self.lazy_time = 1
        self.time = 0

    def decide(self, zetas, combinarial_bandit, arm_set=None, **kwargs):
        if combinarial_bandit:
            self.p = self.distr_exp4(k=self.k, zetas=zetas)
        else:
            self.p = self.distr_exp4(arm_set=arm_set, zetas=zetas)

        if self.time % self.lazy_time == 0:
            if combinarial_bandit:
                self.recommend_arm = draw(self.p, k=self.k)
            else:
                self.recommend_arm = draw(self.p, arm_set=arm_set)
        if self.recommend_arm in self.p.keys():
            prob = self.p[self.recommend_arm]
        else:
            prob = self.p["remains"]
        self.time += 1
        return self.recommend_arm, prob
