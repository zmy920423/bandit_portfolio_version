import torch
import numpy as np
import sympy
from collections import OrderedDict
from sklearn.covariance import LedoitWolf, EmpiricalCovariance


def ellipse_confidence(d, lambda_, delta, t, R, S, L=None, A=None):
    """
    :param A: a dXd positive definite matrix
    :param lambda_: lambda_ > 0
    :param delta: delta
    :param t: time
    :param R: eta_t is conditionally R-sub-Gaussian
    :param S: norm of theta*
    :param L: norm of X
    """
    if L:
        alpha = R * torch.sqrt(d * torch.log((1 + t * L * L / lambda_)
                                             / delta)) + S * torch.sqrt(lambda_)
    else:
        alpha = R * torch.sqrt(torch.log(torch.det(A) / torch.pow(lambda_, d) *
                                         (delta * delta))) + S * torch.sqrt(lambda_)

        # a = torch.sqrt(torch.det(A))
        # b = 1.0 / torch.sqrt(torch.det(torch.eye(d)) * lambda_)
        # c = 2 * torch.log(a * b / delta(t))
        # dd = torch.sqrt(c)
        # alpha = R * torch.sqrt(2 * torch.log(torch.sqrt(torch.det(A))) * 1.0 / torch.sqrt(torch.det(torch.eye(d)) * lambda_)
        #                        / delta(t)) + torch.sqrt(lambda_) * S
    return alpha


def sketch_ellipse_confidence(d, lambda_, delta, t, R, S, m, rou, L=None, A=None):
    """
    :param A: a dXd positive definite matrix
    :param lambda_: lambda_ > 0
    :param delta: delta
    :param t: time
    :param R: eta_t is conditionally R-sub-Gaussian
    :param S: norm of theta*
    :param L: norm of X
    """
    alpha = R * torch.sqrt(
        m * torch.log((1 + t * L * L / (m * lambda_))) + 2 * torch.log(torch.tensor(1) / delta) + d * torch.log(
            1 + rou / lambda_)) * torch.sqrt(1+ rou / lambda_) + S * torch.sqrt(lambda_) * torch.sqrt(1+ rou / lambda_)

    return alpha


def q_learning_confidence(lambda_, t, q=torch.tensor(0.008), e=torch.tensor(0.001)):
    """
    :param q: the convergence rate of q-learning
    :param e: the error of q-learning
    :param lambda_: lambda_ > 0
    :return: confidence bound of q-learning
    """
    confidence_q = torch.tensor(2) * (q + e) * (torch.tensor(1) - torch.pow((q + e), t)) \
                   / (torch.sqrt(lambda_) * (torch.tensor(1) - (q + e)))
    return confidence_q


def hLinUCB_confidence(C, lambda_, delta, t, R, S, L):
    """
    :param C: a lXl positive definite matrix
    :param lambda_: lambda_ > 0
    :param delta_func: delta
    :param t: time
    :param R: eta_t is conditionally R-sub-Gaussian
    :param S: norm of theta*
    :param L: norm of X
    :return alpha: confidence bound of hidden dimension 'v'
    """
    # l = len(C)
    # delta_t = delta_func(t)
    # alpha = R * torch.sqrt(l * torch.log((1 + t * L * L / lambda_)
    #                                      / delta_t)) + S * torch.sqrt(lambda_) + q_learning_confidence(lambda_, t)

    alpha = ellipse_confidence(C, lambda_, delta, t, R, S, L) + q_learning_confidence(lambda_, t)
    return alpha


def wqy_confidence_ellipse(A, lambda_, delta, t, R, S, L=None):
    alpha = 0.1 * torch.sqrt(torch.log(t + 1))
    return alpha


def wqy_hidden_confidence_ellipse(A, lambda_, delta, t, R, S, L=None):
    alpha = 0.1 * torch.sqrt(torch.log(t + 1)) + 0.1 * (1 - 0.8 ** t)
    return alpha


def mutualTag_confidence(A, L1, L2, L5, lambda_, delta, t, R):
    Q2 = 0.01 * (1 - 0.8 ** t)
    Q5 = 0.01 * (1 - 0.8 ** t)
    d = len(A)
    term1 = L1 * L2 * L2 * Q2 / torch.sqrt(lambda_)
    term2 = L1 * L5 * L5 * Q5 / torch.sqrt(lambda_)
    # term3 = 1 + torch.sqrt(d * torch.log( 1 + t * torch.sqrt(L2 * L2 + L5 * L5) / lambda_) / delta)
    det_A = torch.Tensor([np.linalg.det(A.numpy())])
    term3 = torch.sqrt(2 * R * torch.log(torch.sqrt(det_A)/ (lambda_ * delta)))
    term4 = torch.sqrt(lambda_) * L1 * 0.5
    alpha = term1 + term2 + term3 + term4
    return alpha


def draw(prob, arm_set=None, k=None):
    """
    when arm_set is not None: density
    when k is not None: sparsity (for combinarial bandit)
    Input:  prob = dict
            prob[idx_combinations] = value
            prob['remains'] = remaining p value
    """
    # compute total prob
    if arm_set is not None:
        the_sum = sum(prob.values())
    else:
        the_sum = (k - len(prob)) * prob['remains']
        for item in prob.items():
            the_sum += item[1]
    # random a choice
    choice = torch.Tensor(1).uniform_(0, the_sum)[0]
    # get recommend index
    if arm_set is not None:
        for arm_idx, probability in prob.items():
            choice -= probability
            if choice <= 0:
                # 返回的是最后一轮的S作为子集和权值x,就是更新之后的第t+1轮
                return arm_idx
    else:
        pre_idx = 0
        remain = prob['remains']
        prob.pop('remains')
        for arm_idx in sorted(prob.keys()):
            arm_idx = float(arm_idx)
            if choice < torch.tensor(arm_idx - pre_idx - 1.) * remain:
                prob['remains'] = remain
                return int(choice / remain) + pre_idx
            choice -= torch.tensor(arm_idx - pre_idx - 1.) * remain
            choice -= prob[arm_idx]
            if choice <= 0.:
                prob['remains'] = remain
                return int(arm_idx)
            pre_idx = arm_idx

        prob['remains'] = remain
        return int(choice / remain) + pre_idx


def gen_mask_via_index(s_idx, k, d):
    """
    根据s的index生成d维的掩码
    :param s_idx: s为在k基数约束下的组合掩码对应的值, s_idx为第idx个s, 从0开始
    :param k: k基数约束
    :param d: 目标维度
    :return: s_mask向量(d)
    """
    s_idx += 1
    mask_s = torch.zeros(d)
    for i in range(d, -1, -1):
        t = sympy.binomial(i - 1, k)
        if t < s_idx:
            mask_s[d - i] = 1
            s_idx -= t
            k -= 1
        if k <= 0 or s_idx <= 0:
            break
    return mask_s


def gen_mask_via_select_arms(selected_arms, d):
    """
    根据select_arms生成d维的掩码
    :param selected_arms: 应该为1的arm集合
    :param d: 目标维度
    :return: s_mask向量(d)
    """
    mask_s = torch.zeros(d)
    for i in range(d, -1, -1):
        if i in selected_arms:
            mask_s[i] = 1
    return mask_s


def gen_index_with_mask(s_mask, node_size):
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


def zip_portfolio(mask_s,stock_return, portfolio, risk=False, covariance=None):
    x_s_pos = mask_s != 0
    stock_return = torch.tensor(list(stock_return))
    comp_stock_return = stock_return[x_s_pos]  # 取出需要优化的那k个被选择的资产的收益率
    portfolio_reward = torch.dot(portfolio, stock_return)
    last_weight = portfolio[x_s_pos]
    last_weight_hat = last_weight * stock_return[x_s_pos] / portfolio_reward
    if risk:
        comp_covariance = []
        for line in range(len(x_s_pos)):
            if x_s_pos[line]:
                tmp_list = list(covariance[line][x_s_pos])
                comp_covariance.append(tmp_list)
        comp_covariance = torch.tensor(comp_covariance)
        return comp_stock_return,comp_covariance, portfolio_reward, last_weight, last_weight_hat
    else:
        return comp_stock_return,portfolio_reward,last_weight,last_weight_hat


def unzip_portfolio(mask_s, arm_keys, weight):
    i = 0
    portfolio = OrderedDict()
    for stock in arm_keys:
        if mask_s[int(stock)] == 1:
            if len(weight) == 1:
                portfolio[str(stock)] = weight
            else:
                portfolio[str(stock)] = weight[i]
                i += 1
        else:
            portfolio[str(stock)] = torch.tensor(0.)
    return portfolio


def fast_inverse(A_inverse, x): # x为一个向量
    # 可用于 ridge regression 相关算法
    # A += np.ger(x, x)时，可用于快速转置
    xxT = torch.ger(x, x)
    d2 = x.unsqueeze(0).mm(A_inverse).mv(x) + 1 + 1e-9
    d1 = A_inverse.mm(xxT).mm(A_inverse)
    next_A_inverse = A_inverse - torch.div(d1, d2)
    return next_A_inverse


def fast_inverse_multix(A_inverse, x): # x为n个d维度向量拼接 size:(n,d)
    # 可用于 ridge regression 相关算法
    # A += np.ger(x, x)时，可用于快速转置
    mid = torch.eye(len(x)) + x.mm(A_inverse).mm(x.t()) + 1e-9
    left = A_inverse.mm(x.t())
    right = x.mm(A_inverse)
    next_A_inverse = A_inverse - left.mm(torch.inverse(mid)).mm(right)
    return next_A_inverse

    #Sample co-variance
def estimator_sample(returns):
    cov = EmpiricalCovariance().fit(returns).covariance_ #centers the data
    return torch.tensor(cov).float()

    #Ledoit-Wolf shrinkage co-variance estimator
def estimator_ledoit_wolf(returns):
    cov = LedoitWolf().fit(returns).covariance_ #centers the data
    return torch.tensor(cov).float()




if __name__ == '__main__':
    a = gen_index_with_mask(np.array([1,1,1,0]), 4)
    print(a)
    b = torch.tensor(5)
    # b = b.numpy()
    b = bin(int(b))[2:]
    a = torch.rand(6)
    b = torch.flip(a, [0])
    a = torch.pow(2, torch.tensor(100))
    b = 2**500
    # b = sympy.binomial(500,9)
    print(a)
    print(b)