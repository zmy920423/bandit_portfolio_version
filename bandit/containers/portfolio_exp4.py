from bandit.containers.container import Container
import torch as torch
from utils.data import BanditData
from bandit.modules.update_zeta_exp4 import UpdateZetaExp4
from collections import OrderedDict
from bandit.functional import gen_mask_via_index, unzip_portfolio, zip_portfolio
import copy


class PortfolioEXP4(Container):
    def __init__(self, container_id=-1, exp_module=None, module={"file": "eg", "name": "EG"}, **kwargs):
        if exp_module is None:
            super(PortfolioEXP4, self).__init__(container_id=container_id, module=UpdateZetaExp4(container_id))
        else:
            import_str = "from bandit.modules.{} import {}".format(exp_module["file"], exp_module["name"])
            exec(import_str)
            exp = eval(exp_module["name"] + "(**kwargs)")
            super(PortfolioEXP4, self).__init__(container_id=container_id, module=exp)
        import_str = "from bandit.modules.{} import {}".format(module["file"], module["name"])
        exec(import_str)
        self.sub_module_templete = eval(module["name"] + "(**kwargs)")
        self.k = kwargs["k"]
        self.arms = kwargs["arms"]
        self.recommend_arm = 0
        self.prob = 0

    def decide(self, bandit_data):
        # pick an assets combination
        if self.prob == 0:
            self.recommend_arm, self.prob = self._module.decide(bandit_data=bandit_data, combinarial_bandit=True)
        portfolio = self.__get_portfolio(self.recommend_arm, bandit_data.arm_reward)
        self.recommend_arm, self.prob = self._module.decide(bandit_data=bandit_data, combinarial_bandit=True)
        res = {"portfolio": portfolio, "recommend_arm": self.recommend_arm, "prob": self.prob}
        return res

    def update(self, result, bandit_data):
        recommend_arm = result["recommend_arm"]
        prob = result["prob"]
        portfolio = torch.tensor(list(result["portfolio"].values()))
        # 压缩数据
        self.__update_one_arm(recommend_arm, portfolio, bandit_data.arm_reward, prob)

    def __get_portfolio(self, recommend_arm, arm_reward):
        mask_s = gen_mask_via_index(recommend_arm, self.k, len(arm_reward))

        # compute weight
        if recommend_arm not in self._sub_modules.keys():
            sub_module = copy.deepcopy(self.sub_module_templete)
            sub_module.set_id(recommend_arm)
            sub_module.init(mask_s[mask_s != 0].shape[0])
            self.add_sub_module(sub_module)
        weight = self._sub_modules[recommend_arm].decide()["w"]

        # 解压缩
        portfolio = unzip_portfolio(mask_s, arm_reward.keys(), weight)
        return portfolio

    def __update_one_arm(self, recommend_arm, portfolio, arm_reward, prob=1):
        # 压缩数据
        mask_s = gen_mask_via_index(recommend_arm, self.k, len(arm_reward))
        comp_stock_return, portfolio_reward, last_weight, last_weight_hat = zip_portfolio(mask_s, arm_reward.values(),
                                                                                          portfolio)

        # compute weight
        if recommend_arm not in self._sub_modules.keys():
            sub_module = copy.deepcopy(self.sub_module_templete)
            sub_module.set_id(recommend_arm)
            sub_module.init(mask_s[mask_s != 0].shape[0])
            self.add_sub_module(sub_module)
        self._sub_modules[recommend_arm](arm_reward=comp_stock_return, last_weight=last_weight,
                                         last_weight_hat=last_weight_hat, prob=prob)
        if isinstance(self._module, Container):
            self._module.update(recommend_arm, portfolio_reward)
        else:
            self._module(recommend_arm, portfolio_reward)


if __name__ == '__main__':
    exp4 = PortfolioEXP4(k=1, arms=3)
    bandit_data = BanditData(arm_reward={'0': torch.tensor(0.9), '1': torch.tensor(1.2)},
                             arm_context={'1': torch.tensor(-1)})
    print(exp4(bandit_data))
