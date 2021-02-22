import os
import sys
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PARENT_DIR)
from run.portfolio.run_portfolio import evaluator_config, datasets_config
from bandit.containers.portfolio_exp4 import PortfolioEXP4
import sympy
import numpy as np
import torch


def parameter_config(exp_module, module, paras, methods={}, paras_exec=""):
    if not paras:
        methods["new" + exp_module["name"] + "." + module["name"] + "_" + paras_exec] = eval(
            "PortfolioEXP4(exp_module=" + str(exp_module) + ", module=" + str(module) + " ," + paras_exec + ")")
        return methods
    para_name = list(paras.keys())[0]
    paras_new = paras.copy()
    para_values = paras_new.pop(para_name)
    for para in para_values:
        paras_exec_new = paras_exec + para_name + "=" + str(para) + ","
        methods = parameter_config(exp_module, module, paras_new, methods, paras_exec_new)
    return methods


def methods_config_para_turn(exp_modules, modules):
    methods = {}
    for exp_module in exp_modules:
        for module in modules:
            paras = dict(exp_module["para"], **module["para"])
            methods = parameter_config(exp_module, module, paras, methods)
    return methods


def run(datasets, methods, regrets=None, times=None, portfolios=None,  train_size=12):
    for dataset_name, dataset in datasets.items():
        data_round = 0
        for dataset_batch in dataset:
            data_reader = dataset_batch
            batch_data = data_reader.fetch_next_batch(batch_size=100)
            while batch_data is not None:
                for bandit_data in batch_data:
                    if data_round < train_size:
                        data_round += 1
                        continue
                    for method_name, method in methods.items():
                        portfolio = method(bandit_data)["portfolio"]
                        portfolios[dataset_name + "_" + method_name](portfolio, bandit_data)
                        times[dataset_name + "_" + method_name](method.get_time())
                batch_data = data_reader.fetch_next_batch(batch_size=100)
        for method_name, method in methods.items():
            times[dataset_name + "_" + method_name].save(["time", "acc_time"])
            portfolios[dataset_name + "_" + method_name].final_cm()
            portfolios[dataset_name + "_" + method_name].after_eval()
            # portfolios[dataset_name + "_" + method_name].save(["cm_reward", "turnovers"])
            portfolios[dataset_name + "_" + method_name].save(["cm_reward", "net_reward", "weight", "turnovers", "volatility", "sharpe_ratio", "max_draw", "calmar_ratio"])
            print("finish")


def real_data_config(tc):
    print(tc)

    datasets_dir = {"nyse": "/data/portfolio_relative/nyse"}  # 36
    dataset = {"k": 36, "total_round": 5651, "g_max": 1.4}

    # cardinality = dataset["k"]
    cardinality = 5
    transaction_cost = tc
    tcgp_paras = {"k": [cardinality], "eta": [0.001], "gamma": [tc, tc*10]}

    module = [{"file": "tcgp", "name": "TCGP", "para": tcgp_paras}, ]
    exp4_paras = {"total_round": [dataset["total_round"]],
                  "arms": [int(sympy.binomial(dataset["k"], cardinality))],
                  "g_max": [dataset["g_max"]], "f": [5]}
    exp_module = [{"file": "update_zeta_exp4", "name": "UpdateZetaExp4", "para": exp4_paras}]

    methods = methods_config_para_turn(exp_module, module)
    datasets = datasets_config(datasets_dir)
    regrets, times, portfolios = evaluator_config(methods, datasets, transaction_cost)
    train_size = dataset['train_size'] if 'train_size' in dataset else 12
    run(datasets, methods, regrets, times, portfolios, train_size)


if __name__ == '__main__':
    # real_data_config(0)
    # for tc in np.linspace(0, 0.01, 11):
    #     real_data_config(tc)
    real_data_config(0.005)