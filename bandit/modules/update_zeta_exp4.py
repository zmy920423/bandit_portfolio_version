from bandit.containers.container import Container
from bandit.modules.exp4 import Exp4LossBased,Exp4,LazyExp4
from bandit.modules.zeta import ZetaHeap
import torch as torch
import copy
from utils.data import BanditData
from multiprocess.pool import Pool
from config.constants import IDX_M
import os


class UpdateZetaExp4(Container):
    """
    作为在线更新zeta的exp4模型的容器，根结点是exp4，leaf是zeta生成器，一共2^F个arm
    """
    def __init__(self, container_id=-1, f=1, arms=1, k=None, **kwargs):
        m = torch.pow(2, torch.tensor(f)) - 1
        super(UpdateZetaExp4, self).__init__(container_id=container_id, module=LazyExp4(container_id, m=m, arms=arms, **kwargs))
        for i in range(int(m)):
            zeta = ZetaHeap(module_id=i, f=f, arms=arms, k=k)
            self.add_sub_module(zeta)
        self.zetas = []
        self.arm_context = None
        self.arm_reward = None

    def decide(self, bandit_data, combinarial_bandit=True):
        # convert arm context to tensor
        self.arm_context = None
        self.arm_reward = torch.tensor(list(bandit_data.arm_reward.values()))
        for context in bandit_data.arm_context.values():
            if self.arm_context is None:
                self.arm_context = context.view(1, context.shape[0])
            else:
                self.arm_context = torch.cat((self.arm_context,  context.view(1, context.shape[0])), 0)
        self.zetas = self.__gen_zetas()
        recommend_arm, prob = self._module.decide(zetas=self.zetas, combinarial_bandit=combinarial_bandit)
        return recommend_arm, prob

    def update(self, result, reward):
        """
        不规范更新，超过1个参数，所以不能调用call，必须调用update
        :param result: 选择的组合
        :param reward: 组合的return
        :return:
        """
        self._module(result, reward, self.zetas)

    def __gen_zetas(self):
        num_proc = os.cpu_count()
        pool = Pool(processes=num_proc)
        idx = self._sub_modules.keys()
        # zetas = []
        # 生成专家向量
        # for i in idx:
        #     zetas.append(self.gen_zetas_step(i))
        zetas = pool.map(self.gen_zetas_step, idx)
        pool.close()
        pool.join()
        return zetas

    def gen_zetas_step(self, idx):
        # 每个zeta是一个m+1个值的dict，idx_m:1/rank, remains:剩下的平均值
        if IDX_M > 100:
            context = copy.deepcopy(self.arm_reward)
        else:
            context = copy.deepcopy(self.arm_context)
        zeta = self._sub_modules[idx].decide(context)
        self._sub_modules[idx].update(self.arm_context, self.arm_reward)
        return zeta


if __name__ == '__main__':
    LEXP4 = UpdateZetaExp4(container_id=-1, f=3, arms=3, k=2)
    bandit_data = BanditData(arm_reward={'0': torch.tensor(0.9), '1': torch.tensor(1.2), '2': torch.tensor(1.2)},
                             arm_context={'0': torch.tensor([7,8,9]), '1': torch.tensor([1,2,3]), '2': torch.tensor([4,5,6])})
    a = LEXP4.decide(bandit_data)
    print(a)