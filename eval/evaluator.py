from collections import OrderedDict
import pandas as pd
import torch
from utils.data import BanditData
from config.constants import EVAL_DISPLAY_BATCH
import os


class Evaluator(object):
    def __init__(self, file_dir, method="Default", dataset="Default"):

        is_exists = os.path.exists(file_dir)
        if not is_exists:
            os.makedirs(file_dir)
        self._file_dir = file_dir

        self._metrics = OrderedDict()
        self._method = method
        self._dataset = dataset
        self._times = 0

    def eval(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self._times += 1
        self.eval(*args, **kwargs)

    def save(self, keywords):
        """
        save metrics by metric name
        :param keywords: name of metrics
        """
        file_name = self._file_dir + '/{dataset}_{keyword}_{method}.csv'
        for keyword in keywords:
            if not isinstance(self._metrics[keyword],list):
                metric = pd.DataFrame([self._metrics[keyword]])
            else:
                metric = pd.DataFrame(self._metrics[keyword])
            metric.to_csv(file_name.format(dataset=self._dataset, method=self._method, keyword=keyword), index=True, sep=',')
            
    def register_metric(self, name, metric):
        if '_metrics' not in self.__dict__:
            raise AttributeError(
                "cannot assign metric before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("metric name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("metric name can't contain \".\"")
        elif name == '':
            raise KeyError("metric name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._metrics:
            raise KeyError("attribute '{}' already exists".format(name))

        if metric is None:
            self._metrics[name] = None
        elif not (isinstance(metric, list) or isinstance(metric, torch.Tensor)):
            raise TypeError("cannot assign '{}' object to list '{}' "
                            "(list or None required)"
                            .format(type(metric), name))
        else:
            self._metrics[name] = metric

    def __getattr__(self, name):
        if '_metrics' in self.__dict__:
            _metrics = self.__dict__['_metrics']
            if name in _metrics:
                return _metrics[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        metrics = self.__dict__.get('_metrics')
        if isinstance(value, list) or isinstance(value, torch.Tensor):
            if metrics is None:
                raise AttributeError(
                    "cannot assign metrics before Evaluator.__init__() call")
            self.register_metric(name, value)
        elif metrics is not None and name in metrics:
            if value is not None:
                raise TypeError("cannot assign '{}' as metric '{}' "
                                "(list or None expected)"
                                .format(torch.typename(value), name))
            self.register_metric(name, value)
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._metrics:
            del self._metrics[name]
        else:
            object.__delattr__(self, name)


class RegretEval(Evaluator):
    def __init__(self, file_dir, method="Default", dataset="Default", regret_type="Pseudo"):

        super().__init__(file_dir, method, dataset)
        self.regret = []
        self.regret_type = regret_type
        self.max_arm = None
        self.theta = []
        self.arm_times_dict = OrderedDict()
        self.arm_times = []

    def set_max_arm(self, max_arm):
        self.max_arm = max_arm

    def update_regret(self, result, bandit_data):
        if self.regret_type == "Pseudo":
            regret = torch.max(torch.tensor(list(bandit_data.arm_true_reward.values()))) - bandit_data.arm_true_reward[str(result)]
        elif self.regret_type == "True":
            regret = torch.max(torch.tensor(list(bandit_data.arm_reward.values()))) - bandit_data.arm_reward[str(result)]
        elif self.regret_type == "Weak":
            if self.max_arm is None:
                raise AttributeError("cannot compute weak regret before max_arm setting.")
            regret = bandit_data.arm_reward[str(self.max_arm)] - bandit_data.arm_reward[str(result)]
        else:
            raise TypeError("cannot compute regret by {} type.".format(self.regret_type))
        if len(self.regret) == 0:
            self.regret.append(regret)
        else:
            self.regret.append(self.regret[-1] + regret)
        if self._times % EVAL_DISPLAY_BATCH == 0:
            acc_regret = self.regret[-1]
            # print('regret for {} after {} iterations is: {}.'.format(self._method, self._times, acc_regret))

    def update_theta(self, theta):
        self.theta.append(theta)

    def eval(self, result, theta, bandit_data):
        if not isinstance(bandit_data, BanditData):
            raise TypeError("cannot assign '{}' object to BanditData object.".format(type(bandit_data)))
        self.update_regret(result, bandit_data)
        self.update_theta(theta)
        self.update_arm_times(result)

    def update_arm_times(self, result):
        if result not in self.arm_times_dict.keys():
            self.arm_times_dict[result] = 1
        else:
            self.arm_times_dict[result] += 1
        self.arm_times = list(self.arm_times_dict.values())


class TimeEval(Evaluator):
    """
    Time Evaluator to eval running time

    :param self.time: running time of per round
    :param self.acc_time: total running time
    """
    def __init__(self, file_dir, method="Default", dataset="Default"):
        super().__init__(file_dir, method, dataset)
        self.time = [0.]
        self.acc_time = torch.tensor(0.)

    def update_time(self, call_time):
        self.time.append(call_time.total_seconds())
        self.acc_time += torch.tensor(self.time[-1])

    def eval(self, call_time):
        """
        eval function
        :param call_time: call time
        :return:
        """
        self.update_time(call_time)



