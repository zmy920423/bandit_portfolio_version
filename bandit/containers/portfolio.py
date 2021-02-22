from bandit.containers.container import Container
import torch as torch
from collections import OrderedDict
from bandit.functional import estimator_sample,estimator_ledoit_wolf

class Portfolio(Container):
    """
    portfolio class
    """

    def __init__(self, container_id=-1, module={"file": "eg", "name": "EG"}, **kwargs):
        if module["name"] in ["OPAMC","ROPATC"]:
            self.risk = True
            self.returns = []
        else:
            self.risk = False
            self.returns = None
        self.k = kwargs["k"]

        import_str = "from bandit.modules.{} import {}".format(module["file"], module["name"])
        exec(import_str)
        module = eval(module["name"] + "(**kwargs)")
        super(Portfolio, self).__init__(container_id=container_id, module=module)

    def decide(self, bandit_data):
        portfolio = OrderedDict()
        weight = self._module.decide()["w"]
        i = 0
        for stock in bandit_data.arm_reward.keys():
            portfolio[stock] = weight[i]
            i += 1
        return portfolio

    def update(self, result, bandit_data, **kwargs):
        stock_return = torch.tensor(list(bandit_data.arm_reward.values()))
        last_weight = torch.tensor(list(result.values()))
        last_weight_hat = last_weight * stock_return / torch.dot(last_weight, stock_return)
        arm_context = bandit_data.arm_context
        if self.risk:
            self.returns.append(stock_return.numpy().tolist())
            covariance = estimator_ledoit_wolf(self.returns)
            self._module(mean=stock_return, last_weight=last_weight, last_weight_hat=last_weight_hat, covariance=covariance)
        else:
            self._module(arm_reward=stock_return, last_weight=last_weight, last_weight_hat=last_weight_hat, arm_context = arm_context)


class PortfolioCardinality(Container):
    """
    portfolio class
    """

    def __init__(self, container_id=-1, module={"file": "eg", "name": "EG"}, **kwargs):
        if module["name"] in ["OPAMC","ROPATC"]:
            self.risk = True
            self.returns = []
        else:
            self.risk = False
            self.returns = None
        self.k = kwargs["k"]
        import_str = "from bandit.modules.{} import {}".format(module["file"], module["name"])
        exec(import_str)
        module = eval(module["name"] + "(**kwargs)")
        super(PortfolioCardinality, self).__init__(container_id=container_id, module=module)

    def decide(self, bandit_data):
        portfolio = OrderedDict()
        weight = self._module.decide()["w"]
        idx = torch.topk(weight, self.k).indices
        weight_c = torch.zeros(len(weight))
        weight_c[idx] = weight[idx]
        weight_c = weight_c / torch.sum(weight_c)
        i = 0
        for stock in bandit_data.arm_reward.keys():
            portfolio[stock] = weight_c[i]
            i += 1
        return portfolio

    def update(self, result, bandit_data, **kwargs):
        stock_return = torch.tensor(list(bandit_data.arm_reward.values()))
        last_weight = torch.tensor(list(result.values()))
        last_weight_hat = last_weight * stock_return / torch.dot(last_weight, stock_return)
        arm_context = bandit_data.arm_context
        if self.risk:
            self.returns.append(stock_return.numpy().tolist())
            covariance = estimator_ledoit_wolf(self.returns)
            self._module(mean=stock_return, last_weight=last_weight, last_weight_hat=last_weight_hat, covariance=covariance)
        else:
            self._module(arm_reward=stock_return, last_weight=last_weight, last_weight_hat=last_weight_hat, arm_context = arm_context)

