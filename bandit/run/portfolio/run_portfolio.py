import os
import sys
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PARENT_DIR)
import torch
import numpy as np
from utils.data.data_reader import DataReader
from eval.evaluator import TimeEval
from eval.portfolio_eval import PortfolioEval, PortfolioRegretEval
from bandit.containers.portfolio import Portfolio

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

REGRET_TYPE = "OLU_BCRP"


def parameter_config(module, paras, methods={}, paras_exec=""):
    if not paras:
        methods[module["name"] + "_" + paras_exec] = eval("Portfolio(module=" + str(module) + " ," + paras_exec + ")")
        return methods
    para_name = list(paras.keys())[0]
    paras_new = paras.copy()
    para_values = paras_new.pop(para_name)
    for para in para_values:
        paras_exec_new = paras_exec + para_name + "=" + str(para) + ","
        methods = parameter_config(module, paras_new, methods, paras_exec_new)
    return methods


def methods_config_para_turn(modules):
    methods = {}
    for module in modules:
        methods = parameter_config(module, module["para"], methods)
    return methods


def datasets_config(datasets_dir):
    datasets = {}
    for dataset_name, dataset_file in datasets_dir.items():
        file_dir = PARENT_DIR + dataset_file
        file_list = os.listdir(file_dir)  # 列出文件夹下所有的目录与文件
        dataset = []
        for i in range(len(file_list)):
            path = os.path.join(file_dir, file_list[i])
            if os.path.isfile(path):
                dataset.append(DataReader(file_path=path))
        datasets[dataset_name] = dataset
    return datasets


def evaluator_config(methods, datasets, transaction_cost, bcrp=None):
    regrets = {}
    times = {}
    portfolios = {}
    for dataset in datasets.keys():
        file_dir = PARENT_DIR + "/result/portfolio/{}_cost_{:.3f}".format(str(dataset), transaction_cost)
        for method_name in methods.keys():
            regret = PortfolioRegretEval(file_dir=file_dir, method=method_name,
                                         dataset=dataset, regret_type=REGRET_TYPE)
            if bcrp is not None:
                regret.set_bcrp(bcrp)
            time = TimeEval(file_dir=file_dir, method=method_name,dataset=dataset)
            portfolio = PortfolioEval(file_dir=file_dir, method=method_name,dataset=dataset)
            portfolio.set_transaction_cost(transaction_cost)
            regrets[dataset + "_" + method_name] = regret
            times[dataset + "_" + method_name] = time
            portfolios[dataset + "_" + method_name] = portfolio
    return regrets, times, portfolios


def run(datasets, methods, regrets, times, portfolios, train_size=12):
    for dataset_name, dataset in datasets.items():
        for dataset_batch in dataset:
            data_round = 0
            data_reader = dataset_batch
            batch_data = data_reader.fetch_next_batch(batch_size=100)
            while batch_data is not None:
                for bandit_data in batch_data:
                    if data_round < train_size:
                        data_round += 1
                        continue
                    for method_name, method in methods.items():
                        portfolio = method(bandit_data)
                        # print(portfolio)
                        portfolios[dataset_name + "_" + method_name](portfolio, bandit_data)
                        # regrets[dataset_name + "_" + method_name](portfolio, bandit_data)
                        times[dataset_name + "_" + method_name](method.get_time())
                batch_data = data_reader.fetch_next_batch(batch_size=100)
        for method_name, method in methods.items():
            # print final result
            portfolios[dataset_name + "_" + method_name].final_cm()
            # regrets[dataset_name + "_" + method_name].save(["regret", "final_regret"])
            times[dataset_name + "_" + method_name].save(["time", "acc_time"])
            portfolios[dataset_name + "_" + method_name].after_eval()
            portfolios[dataset_name + "_" + method_name].save(
                ["cm_reward", "net_reward", "weight", "turnovers", "volatility", "sharpe_ratio", "max_draw", "calmar_ratio"])
        print("finish")


def simulate_config():
    num_round = range(1000, 11000, 1000)
    bcrp = [torch.tensor([2.6063e-01, -2.3909e-12, 2.7857e-12, 1.3523e-01, -1.3018e-12,
                          1.2497e-01, 1.0013e-01, 3.3662e-02, 3.3483e-02, 3.1190e-01]),
            torch.tensor([1.3000e-01, -2.8313e-12, 1.1626e-01, 1.0992e-01, 5.6720e-02,
                          2.0100e-01, -3.9600e-12, 7.2101e-02, 1.1136e-01, 2.0263e-01]),
            torch.tensor([0.1062, 0.0278, 0.0645, 0.2705, 0.1352, 0.0640, 0.1254, 0.0944, 0.0181, 0.0939]),
            torch.tensor([1.6923e-01, 4.0209e-02, 9.6648e-02, -4.1139e-12, 4.4150e-02,
                          1.5421e-01, 1.9049e-01, 1.2322e-01, 1.1930e-01, 6.2550e-02]),
            torch.tensor([0.0135, 0.1187, 0.0650, 0.1389, 0.1073, 0.1943, 0.0959, 0.1324, 0.0738, 0.0601]),
            torch.tensor([1.3138e-01, 9.5795e-02, 1.2236e-01, 1.1969e-01, 7.4497e-02,
                          1.3003e-01, 9.0674e-02, 1.4071e-01, -7.2058e-12, 9.4858e-02]),
            torch.tensor([0.1218, 0.0647, 0.1210, 0.0887, 0.0967, 0.1129, 0.1063, 0.0313, 0.1626,
                          0.0939]),
            torch.tensor([0.1468, 0.1697, 0.1266, 0.1162, 0.0513, 0.0522, 0.1013, 0.1544, 0.0514, 0.0301]),
            torch.tensor([0.0442, 0.1009, 0.1343, 0.0656, 0.0929, 0.0945, 0.0724, 0.1487, 0.1289, 0.1176]),
            torch.tensor([0.1459, 0.1077, 0.1088, 0.0658, 0.1484, 0.0622, 0.0674, 0.0615, 0.1042, 0.1281])]
    i = 0
    for T in num_round:
        datasets_dir = {"cardinality_portfolio_n_10_k_10_T_"+str(T): "/data/sim/cardinality_portfolio_n_10_k_10_T_"+str(T)}
        transaction_cost = 0
        eg_paras = {"k": [10], "eta": [0.05]}
        ons_paras = {"k": [10], "eta": [0.0], "beta": [1.], "gamma": [1. / 8]}

        olu_paras = {"k": [10], "eta": [0.1], "gamma": [0.1]}
        tcgp_paras = {"k": [10], "eta": [0.1], "gamma": [0.1]}
        module = [{"file": "eg", "name": "EG", "para": eg_paras},
                  {"file": "olu", "name": "OLU", "para": olu_paras},
                  {"file": "tcgp", "name": "TCGP", "para": tcgp_paras}]
        methods = methods_config_para_turn(module)
        datasets = datasets_config(datasets_dir)
        regrets, times, portfolios = evaluator_config(methods, datasets, transaction_cost, bcrp[i])
        run(datasets, methods, regrets, times, portfolios)
        i += 1


def real_data_config(tc):

    transaction_cost = tc

    datasets_dir = {"nyse": "/data/portfolio_relative/nyse"}  # 36
    dataset = {"k": 36, "total_round": 5651, "g_max": 1.4}
    datasets_dir = {'tse': '/data/portfolio_relative/tse'}  # 88
    dataset = {"k": 88, "total_round": 1259, "g_max": 1.5}
    datasets_dir = {'djia': '/data/portfolio_relative/djia'}  # 30
    dataset = {"k": 30, "total_round": 507, "g_max": 1.3}
    # datasets_dir = {'sp500': '/data/portfolio_relative/sandp'}  # 476
    # dataset = {"k": 476, "total_round": 1258, "g_max": 1.5}
    # # datasets_dir = {'sp500': '/data/portfolio_relative/sp500'}    # 25
    # dataset = {"k": 25, "total_round": 1258, "g_max": 1.5}
    # datasets_dir = {'sp_500_classify_60epochs': '/data/portfolio_relative/sp500_news/feature_classification_60epochs'}  # 478
    # dataset = {"total_round": 255, 'k': 478, 'train_size': 1529, 'data_name': ['sp500'], 'epochs': 200, 'feature_type': ['classify'], 'n_features': 5}  # 1529
    # datasets_dir = {'djia_classify_200epochs': '/data/portfolio_relative/djia_news/feature_classify_200epochs'}  # 29
    # dataset = {"total_round": 299, 'k': 29, 'train_size': 1690, 'data_name': ['djia'], 'epochs': 200, 'feature_type': ['classify'], 'n_features': 5} # 1690


    # eg_paras = {"k": [36], "eta": [0.05]}
    # ons_paras = {"k": [36], "eta": [0.0], "beta": [1.], "delta": [1. / 8]}

    # mvo_paras = {'k': [dataset['k']], 'data_name': [dataset['data_name']], 'epochs': [dataset['epochs']],
    #              'feature_type': [dataset['feature_type']], 'stage': [5, 10], 'n_features': [dataset['n_features']]}
    # olu_paras = {"k": [30], "eta": [10, 1, 0.1, 0.001, ], "gamma": [tc]}
    # tco_paras = {'k':  [30], 'lambda_': [tc], 'eta': [0.1]}
    # drp_paras = {'k':  [30]}

    # module = [{"file": "ons", "name": "ONS", "para": ons_paras},
    #           {"file": "eg", "name": "EG", "para": eg_paras}]
    #           {"file": "tco", "name": "TCO", "para": tco_paras}]
    # module = [{"file": "olu", "name": "OLU", "para": olu_paras}]
    # module = [{"file": "tco", "name": "TCO", "para": tco_paras}]
    # module = [{"file": "drp", "name": "DRP", "para": drp_paras},
    #           {"file": "olu", "name": "OLU", "para": olu_paras},
    #           {"file": "tco", "name": "TCO", "para": tco_paras}]
    # module = [{"file": "mvo", "name": "MVO", "para": mvo_paras}]
    # module = [{"file": "mvo", "name": "MVONews", "para": mvo_paras}]



    # opamc_paras = {"k": [30],"lambda_":[tc, 10*tc], "alpha":[1,1.5,3,10,50,100], "eta":[0.01, 0.001]}
    # module = [{"file": "opamc", "name": "OPAMC", "para": opamc_paras}]

    ropatc_paras = {"k": [30], "lambda_": [100*tc], "rho": [0.01,0.02,0.05,0.1]}
    module = [{"file": "ropatc", "name": "ROPATC", "para": ropatc_paras}]

    methods = methods_config_para_turn(module)
    datasets = datasets_config(datasets_dir)
    regrets, times, portfolios = evaluator_config(methods, datasets, transaction_cost)

    train_size = dataset['train_size'] if 'train_size' in dataset else 12
    run(datasets, methods, regrets, times, portfolios, train_size)


if __name__ == '__main__':
    # simulate_config()

    for tc in [0.005]:  # np.linspace(0, 0.01, 11):
        real_data_config(tc)