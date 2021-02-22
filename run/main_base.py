import os
import random
from utils.data.data_reader import DataReader
from eval.evaluator import RegretEval, TimeEval

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


def parameter_config(module, sub_module, paras, methods={}, paras_exec=""):
    if not paras:
        import_str = "from bandit.containers.{} import {}".format(module["file"], module["name"])
        exec(import_str)
        methods[module["name"] + "." +sub_module["name"] + ":" + paras_exec] = eval(module["name"] + "( module=" + str(sub_module) + " ," + paras_exec + ")")
        return methods
    para_name = random.choice(list(paras.keys()))
    paras_new = paras.copy()
    para_values = paras_new.pop(para_name)
    for para in para_values:
        paras_exec_new = paras_exec + para_name + "=" + str(para) + ","
        methods = parameter_config(module, sub_module, paras_new, methods, paras_exec_new)
    return methods


def methods_config_para_turn(modules, sub_modules):
    methods = {}
    for module in modules:
        for sub_module in sub_modules:
            paras = dict(module["para"], **sub_module["para"])
            methods = parameter_config(module, sub_module, paras, methods)
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


def evaluator_config(methods, datasets):
    evaluator = {}
    regrets = {}
    times = {}
    for dataset in datasets.keys():
        for method_name in methods.keys():
            regret = RegretEval(file_dir=PARENT_DIR + "/result/sim/OFUL", method=method_name, dataset=dataset,
                                regret_type="Pseudo")
            # regret.set_max_arm(2)
            time = TimeEval(file_dir=PARENT_DIR + "/result/sim/OFUL", method=method_name, dataset=dataset)
            regrets[dataset + "_" + method_name] = regret
            times[dataset + "_" + method_name] = time
    evaluator["regret"] = regrets
    evaluator["time"] = times
    return evaluator


def run(datasets, methods, evaluator):
    for dataset_name, dataset in datasets.items():
        for dataset_batch in dataset:
            data_reader = dataset_batch
            batch_data = data_reader.fetch_next_batch(batch_size=100)
            while batch_data is not None:
                for bandit_data in batch_data:
                    for method_name, method in methods.items():
                        result = method(bandit_data)
                        theta_pre = 0
                        evaluator["regret"][dataset_name + "_" + method_name](result, theta_pre, bandit_data)
                        evaluator["time"][dataset_name + "_" + method_name](method.get_time())
                batch_data = data_reader.fetch_next_batch(batch_size=100)
        for method_name, method in methods.items():
            evaluator["regret"][dataset_name + "_" + method_name].save(["regret"])
            evaluator["time"][dataset_name + "_" + method_name].save(["time", "acc_time"])
        print("finish")


if __name__ == '__main__':
    # module配置
    paras = {"d": [100, ], "total_round": [10000, ], "g_max": [1, ]}
    OFUL = {"file": "oful", "name": "OFUL", "para": paras}
    LinearTS = {"file": "linear_ts", "name": "LinearTS", "para": paras}
    module = [OFUL]
    # sub_module配置
    sub_paras = {"m": [10,]}
    Ridge = {"file": "ridge", "name": "Ridge", "para": sub_paras}
    FD_Sketching = {"file": "fd_sketching", "name": "FDSketching", "para": sub_paras}
    Iterative_Sketching = {"file": "iterative_sketching", "name": "IterativeSketching", "para": sub_paras}
    sub_module = [FD_Sketching, Ridge, Iterative_Sketching]
    # dataset配置
    datasets_dir = {"sim": "/data/sim/OFUL"}

    # 动态加载
    methods = methods_config_para_turn(module, sub_module)
    datasets = datasets_config(datasets_dir)
    evaluator = evaluator_config(methods, datasets)

    # 运行
    run(datasets, methods, evaluator)
