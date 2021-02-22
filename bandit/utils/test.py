import sympy
import torch


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


def gen_mask(s_idx, k, d):
    """
            根据s的index生成d维的掩码
            :param s_idx: s为在k基数约束下的组合掩码对应的值, s_idx为第idx个s, 从0开始
            :param k: k基数约束
            :param d: 目标维度
            :return: s_mask向量(d)
            """
    s_idx += 1
    s_mask = torch.zeros(d)
    for i in range(d, 0, -1):
        t = sympy.binomial(i - 1, k)
        if t < s_idx:
            s_mask[d - i] = 1
            s_idx -= t
            k -= 1
        if k <= 0 or s_idx <= 0:
            break
    return s_mask

d=10
k = 2
idx = 7
a = gen_mask(idx,k,10)
print(a)
s = gen_index(a,k)
print(s)