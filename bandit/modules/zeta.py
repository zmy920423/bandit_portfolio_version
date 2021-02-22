import copy
import heapq

import sympy
import torch

from bandit.modules.module import Module
from bandit.parameter import Parameter
from config.constants import IDX_M


class ZetaHeap(Module):
    """
    由堆产生的专家向量，每个向量代表一个专家对所有combination的选择概率建议
    """
    def __init__(self, module_id, f, arms, k=None, lambda_=0.1):
        """
        :param module_id: 专家的编号，也代表了它选择的特征的掩码
        :param f: 一共有多少个特征
        :param arms: 一共多少个arm，来判断m的值 (m是最大保留的arm的数量)
        :param k: arm是在k约束下，如果None表示没有约束
        :param lambda_: ridge regression参数，一般使用默认值
        """
        super(ZetaHeap, self).__init__(module_id=module_id)
        # 专家的mask生成，一共M=2^F个
        self.f = torch.tensor(f)
        self.s_mask = self.__gen_mask(self.get_id(), f)
        d = str(self.s_mask.tolist()).count("1")
        self.s_mask = (self.s_mask != 0)

        self.k = torch.tensor(k)

        # ridge regression参数
        self.theta = Parameter(torch.zeros(d))
        self.A = Parameter(torch.eye(d) * lambda_)
        self.A_inv = Parameter(torch.inverse(self.A))
        self.b = Parameter(torch.zeros(d))

        # 求m，使得m以后是平均分布，且delta = 10^-5
        minor = torch.tensor(0.0001)
        self.m = min(self.__get_k_by_delta(minor), arms - 1)
        # 只保留最好的一只组合
        self.m = 1

    def update(self, context, reward):
        """
        利用ridge regression更新特征拟合而来的系数，用于生成之后的zeta
        :param arm_context: 某个stock的context
        :param arm_reward: 某个stock的reward
        :return:
        """
        # 压缩信息，训练context信息
        context = context[:, self.s_mask].float()
        self.A += context.T.mm(context)
        count = reward.shape[0]
        for i in range(count):
            self.b += reward[i] * context[i]
        self.A_inv = torch.inverse(self.A)
        self.theta = self.A_inv.mv(self.b)
        return

    def decide(self, context):
        """
        step1: 预测reward; step2: 利用堆生成zeta
        :param arm_context: 所有stock的context (dict)
        :return:
        """
        # context = context[:, self.s_mask].float()
        # reward_pred = context.mv(self.theta)

        if self.get_id() <= IDX_M:
            context = context[:, self.s_mask].float()
            reward_pred = context.mv(self.theta)
        else:
            reward_pred = context

        return self.__gen_zeta(reward_pred, self.m, self.k)

    @staticmethod
    def __gen_zeta(d, m, node_size):
        """
        把所有组合的元素的index看作一棵树，其中子节点大于本节点的index
        维护一个堆，每个节点是(-mean, src_idx的list）,负的原因是为了每次pop最大值
        每次pop后push两个点，一个是长度加1的点node_long，一个是长度不变最有一个元素+1的点node_width
        如果个数超过node_size则结束
        :param d: 一个数组
        :param m: 前m个子序列
        :param node_size: 最长可以有多少个元素
        :return: 前m个子序列的index
        """
        zeta = {}
        d_sort, idx = torch.sort(d, descending=True)

        max_mean = (-d_sort[0], [0, ])
        heap = []
        heapq.heappush(heap, max_mean)
        k = 1

        while k < m + 1:
            # pop最大值
            max_mean = heapq.heappop(heap)

            if len(max_mean[1]) == node_size:
                # 转换成zeta
                s_idx = torch.zeros(idx.shape[0])
                for src_idx_value in max_mean[1]:
                    s_idx[idx[src_idx_value]] = 1

                def gen_index(s_mask, node_size):
                    s = 0
                    node_k = 1
                    d = s_mask.shape[0]
                    for i in range(1, d + 1):
                        if s_mask[d - i] == 1:
                            s += sympy.binomial(i - 1, node_k)
                            node_k += 1
                        if node_k > node_size:
                            break
                    return s

                zeta[gen_index(s_idx, node_size)] = 1. / k
                k += 1
            if len(max_mean[1]) > node_size:
                continue
            # push新的值
            last_node = max_mean[1][-1]  # 获得最后一个节点在d_sort中的index

            mean = - max_mean[0]
            if last_node < d_sort.shape[0] - 1:
                # 长度加1的点node_long
                mean_long = - (mean * len(max_mean[1]) + d_sort[last_node + 1]) / (len(max_mean[1]) + 1)
                src_idx_long = copy.deepcopy(max_mean[1])
                src_idx_long.append(last_node + 1)
                node_long = (mean_long, src_idx_long)
                heapq.heappush(heap, node_long)
                # 长度不变最有一个元素+1的点node_width
                mean_width = - (mean * len(max_mean[1]) - d_sort[last_node + 0] + d_sort[last_node + 1]) / len(max_mean[1])
                src_idx_width = copy.deepcopy(max_mean[1])
                src_idx_width[-1] = last_node + 1
                node_width = (mean_width, src_idx_width)
                heapq.heappush(heap, node_width)
        zeta["remains"] = 0
        return zeta

    @staticmethod
    def __get_s_idx(tar_idx, scr_idx):
        """
        排序后的转换，生成掩码的s值，和genMask有些些相反，但是只是用于action中zeta生成
        :param tar_idx: 排序对应的顺序，argsort生成
        :param scr_idx: idx的list，代表选择了哪些排序后的资产构成组合
        :return: k
        """
        p_pos = torch.zeros(len(tar_idx))
        for src_idx_value in scr_idx:
            p_pos[tar_idx[src_idx_value]] = 1
        p_pos = 1 * (p_pos != 0)
        p_pos = "".join(map(lambda x: str(x), p_pos))
        p_pos = int(p_pos, 2)
        return p_pos - 1

    @staticmethod
    def __gen_mask(s_idx, d):
        """
        根据s的index生成d维的掩码
        :param s_idx: s为d个元素组合掩码对应的值, s_idx为第idx个s, 从0开始
        :param d: 目标维度
        :return: s_mask向量(d)
        """
        s = s_idx + 1
        s_mask = torch.zeros(d)
        s_byte_list = list(map(int, bin(s).replace('0b', '')))
        for i in range(len(s_byte_list)):
            s_mask[d - i - 1] = s_byte_list[len(s_byte_list) - i - 1]
        return s_mask

    @staticmethod
    def __get_k_by_delta(delta):
        """
        找到前k个，1/k-1/(k+1)< delta，后续的当成平均分布
        :param delta: 很小的一个误差
        :return: k
        """
        k = torch.sqrt(1./delta + 1./4)
        k = k - 0.5
        return int(k)


if __name__ == '__main__':
    a = torch.randint(0, 29, (1,))[0]
    print(a)