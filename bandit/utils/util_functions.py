import json
import numpy as np
from config.constants import BATCH_SIZE
import os
import torch


def normalize(v, s=1):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v * np.sqrt(s) / norm


def obj2json(python_obj):
    """
    将一个class转成json
    :param python_obj: python中自定义的一个class
    :return: json
    """
    return json.dumps(python_obj.__dict__)


def batch_write(logs, data_dir, batch_size=BATCH_SIZE):
    """
    批量写入文件，按照BATCH_SIZE来分割文件
    :param logs: 需要写入的数据
    :param data_dir: 写入的目录
    :return:
    """
    batch_count = 0
    f_bandit_data = open(data_dir + "/bandit_data_"+str(batch_count)+".json", "w")
    for i in range(len(logs)):
        if i % batch_size == 0:
            f_bandit_data.close()
            f_bandit_data = open(data_dir + "/bandit_data_"+str(batch_count)+".json", "w")
            batch_count += 1
        f_bandit_data.write(logs[i]+"\n")


def write_data(logs, arg_name, dir_base, batch_size=BATCH_SIZE):
    """
    写入数据
    :param logs: 需要写入的数据
    :param arg_name: 数据对应的一些必要参数
    :param dir_base: 写入的目录
    :return:
    """
    data_dir = dir_base + '/' + arg_name
    is_exists = os.path.exists(data_dir)
    if not is_exists:
        os.makedirs(data_dir)
    batch_write(logs, data_dir, batch_size)


if __name__ == '__main__':
    v1 = [1.,2.,3.,4.]

