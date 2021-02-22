#-*- coding: utf-8 -*-
from eval.evaluator import Evaluator
from utils.data.data_struct import BanditData
from config.constants import EVAL_DISPLAY_BATCH
import numpy
from scipy.optimize import fsolve
import torch
from collections import OrderedDict


class PortfolioEval(Evaluator):
    def __init__(self, file_dir, method="Default", dataset="Default", frequency=12.):
        super().__init__(file_dir, method, dataset)
        self.cm_reward = [torch.tensor(1.), ]
        self.annualized_percentage_yield = torch.tensor(0.)
        self.net_reward = []
        self.weight = []
        self.weight_o = []
        self.max_draw = torch.tensor(0.)
        self.volatility = torch.tensor(0.)
        self.sharpe_ratio = torch.tensor(0.)
        self.calmar_ratio = torch.tensor(0.)
        self.transaction_cost = torch.tensor(0.)
        self.turnovers = torch.tensor(0.)
        self.frequency = torch.tensor(frequency)

    def set_transaction_cost(self, transaction_cost):
        self.transaction_cost = torch.tensor(transaction_cost)

    def compute_cm(self):
        self.cm_reward.append(self.cm_reward[-1] * self.net_reward[-1])

    def compute_sharpe(self):
        return_std = torch.std(torch.tensor(self.net_reward)) * torch.sqrt(torch.tensor(252.))
        non_risk_rate = torch.tensor(0.)  # 无风险利率为0.02
        self.sharpe_ratio = (self.annualized_percentage_yield - non_risk_rate) / return_std

    def compute_calmar(self):
        self.calmar_ratio = self.annualized_percentage_yield / self.max_draw

    def compute_max_drawdown(self):
        """最大回撤率"""
        i = numpy.argmax((numpy.maximum.accumulate(self.cm_reward) - self.cm_reward) / numpy.maximum.accumulate(self.cm_reward))  # 结束位置
        if i == 0:
            return 0
        j = numpy.argmax(self.cm_reward[:i])  # 开始位置
        self.max_draw = (self.cm_reward[int(j)] - self.cm_reward[int(i)]) / (self.cm_reward[int(j)])

    def compute_volatility(self):
        net_return = torch.tensor(self.net_reward) - torch.tensor(1.)
        return_std = torch.std(net_return)
        self.volatility = torch.sqrt(self.frequency)*return_std

    def compute_turnovers(self):
        self.turnovers /= self._times

    def update_net(self, portfolio, bandit_data):
        w = torch.tensor(list(portfolio.values()))
        if torch.sum(w) == 0:
            net_reward = 1
            self.turnovers += torch.tensor(0)
            w_o = w
        else:
            if len(self.weight_o) == 0:
                w_o = w
            else:
                w_o = self.weight_o[-1]

            stocks = portfolio.keys()
            r = {key: value for key, value in bandit_data.arm_reward.items() if key in stocks}
            r = torch.tensor(list(r.values()))

            def f3(x):
                return numpy.array(x - 1 + self.transaction_cost.numpy() * numpy.sum(numpy.abs(w_o.numpy() - w.numpy() * x)))

            net_proportion = torch.tensor(fsolve(f3, numpy.array(1.))[0],dtype=torch.float32)

            net_reward = torch.dot(w, r) * net_proportion
            self.turnovers += torch.sum(torch.abs(w * net_proportion-w_o))
            w_o = w * r / torch.dot(w, r)


        self.weight.append(list(w.numpy()))
        self.weight_o.append(w_o)
        self.net_reward.append(net_reward)

    def final_cm(self):
        print('final cm for {}_{} is: {}'.format(self._dataset, self._method, self.cm_reward[-1]))

    def eval(self, portfolio, bandit_data):
        if not isinstance(bandit_data, BanditData):
            raise TypeError("cannot assign '{}' object to BanditData object ".format(type(bandit_data)))
        self.update_net(portfolio, bandit_data)
        self.compute_cm()
        acc_cm = self.cm_reward[-1]
        print('cm for {} after {} iterations on date {} is: {}'.format(self._method, self._times, bandit_data.timestamp, acc_cm))
        if self._times % EVAL_DISPLAY_BATCH == 0:
            acc_cm = self.cm_reward[-1]
            print('cm for {} after {} iterations is: {}'.format(self._method, self._times, acc_cm))

    def after_eval(self):
        daily = len(self.cm_reward)
        self.annualized_percentage_yield = torch.pow(self.cm_reward[-1], torch.tensor(252. / daily)) - 1
        self.compute_max_drawdown()
        self.compute_sharpe()
        self.compute_calmar()
        self.compute_volatility()
        self.compute_turnovers()


class PortfolioRegretEval(Evaluator):
    def __init__(self, file_dir, method="Default", dataset="Default", regret_type="OLU_BCRP"):

        super().__init__(file_dir, method, dataset)
        self.regret = []
        self.final_regret = torch.tensor(0)
        self.regret_type = regret_type
        self.cardinality = None
        self.bcrp = None
        self.weight = None
        self.arm_times_dict = OrderedDict()
        self.arm_times = []

    def set_bcrp(self, bcrp_weight):
        object.__setattr__(self, "bcrp", bcrp_weight)
        self.cardinality = bcrp_weight.shape[0]

    def update_regret(self, result, bandit_data):
        if self.regret_type == "BCRP":
            if self.bcrp is None:
                raise AttributeError("cannot compute weak regret before max_arm setting.")
            weight = torch.tensor(list(result.values()))
            stock_return = torch.tensor(list(bandit_data.arm_reward.values()))
            regret = torch.dot(self.bcrp, stock_return[:self.cardinality]) - torch.dot(weight, stock_return)
        elif self.regret_type == "OLU_BCRP":
            if self.bcrp is None:
                raise AttributeError("cannot compute weak regret before max_arm setting.")
            weight = torch.tensor(list(result.values()))
            stock_return = torch.tensor(list(bandit_data.arm_reward.values()))
            if self.weight is None:
                # save hat weight
                object.__setattr__(self, "weight", weight)
            regret = torch.dot(self.bcrp, stock_return[:self.cardinality]) - torch.dot(weight, stock_return) + torch.abs(weight-self.weight).sum()
            object.__setattr__(self, "weight", weight * stock_return / torch.dot(weight, stock_return))
        else:
            raise TypeError("cannot compute regret by {} type.".format(self.regret_type))
        if len(self.regret) == 0:
            self.regret.append(regret)
        else:
            self.regret.append(self.regret[-1] + regret)
        self.final_regret = self.regret[-1]
        if self._times % EVAL_DISPLAY_BATCH == 0:
            print('regret for {} after {} iterations is: {}.'.format(self._method, self._times, self.regret[-1]))

    def eval(self, result, bandit_data):
        if not isinstance(bandit_data, BanditData):
            raise TypeError("cannot assign '{}' object to BanditData object.".format(type(bandit_data)))
        self.update_regret(result, bandit_data)
