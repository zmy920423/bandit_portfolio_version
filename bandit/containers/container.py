from collections import OrderedDict
from ..modules import Module
from utils.data import BanditData
import datetime
import pickle


class Container(object):
    """
    Base class for all bandit modules, which is bandit structure.
    """
    def __init__(self, container_id=-1, module=None, sub_modules=None):
        self.container_id = container_id
        self._sub_modules = OrderedDict()
        self._call_time = 0
        if module is not None:
            if not (isinstance(module, Module) or isinstance(module, Container)):
                raise TypeError("cannot assign '{}' object to Module object "
                                .format(type(module)))
            self._module = module
        else:
            self._module = Module(module_id=-1)
        if sub_modules is not None:
            sub_modules_dict = {}
            for sub_module in sub_modules:
                if not (isinstance(sub_module, Module) or isinstance(sub_module, Container)):
                    raise TypeError("cannot assign '{}' object to Module object "
                                    .format(type(sub_module)))
                sub_modules_dict[sub_module.get_id()] = sub_module
            self._sub_modules.update(sub_modules_dict)

    def get_id(self):
        return self.container_id

    def get_time(self):
        return self._call_time

    def update(self, result, bandit_data, **kwargs):
        raise NotImplementedError

    def decide(self, bandit_data):
        raise NotImplementedError

    def __call__(self, bandit_data):
        self._call_time = datetime.datetime.now()
        if not isinstance(bandit_data, BanditData):
            raise TypeError("cannot assign '{}' object to BanditData object ".format(type(bandit_data)))
        result = self.decide(bandit_data)
        self.update(result, bandit_data)
        self._call_time = datetime.datetime.now() - self._call_time
        return result

    def children(self):
        """Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self._sub_modules.items():
            yield module

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def add_sub_module(self, module):
        if module.get_id() in self._sub_modules.keys():
            raise KeyError("module '{}' already exists".format(module.get_id()))
        else:
            if not (isinstance(module, Module) or isinstance(module, Container)):
                raise TypeError("cannot assign '{}' object to Module object "
                                .format(type(module)))
            self._sub_modules[module.get_id()] = module

    def remove_sub_module(self, module=None, module_id=None):
        if module is not None:
            if module.get_id() in self._sub_modules.keys():
                self._sub_modules.pop(module.get_id())
                return
            else:
                print("Bandit Warning: This arm was already removed.")
                return
        if module_id is not None:
            if module_id in self._sub_modules.keys():
                self._sub_modules.pop(module_id)
                return
            else:
                print("Bandit Warning: This arm was already removed.")
                return

    def save(self, file_folder_path, prev_path=''):
        # if 'container_id' not in self.__dict__:
        #     raise AttributeError(
        #         "cannot assign parameter before {}.__init__() call".format(type(self).__name__))

        content_dict = {}
        if prev_path:
            raise NotImplementedError

        else:
            self_dict = self.__dict__
            file_name = type(self).__name__ + '@' + \
                        str(datetime.datetime.now()).replace('-', '_').replace(' ', '_')
            file_path = file_folder_path + '/' + file_name

            # save _module
            content_dict['_module'] = self._module.save(file_folder_path)

            # save _sub_modules
            content_dict['_sub_modules'] = []
            for each in self._sub_modules:
                if each != -1:
                    content_dict['_sub_modules'].append(self._sub_modules[each].save(file_folder_path))

            # save parameters
            for key in self_dict.keys():
                if key in ['_module', '_sub_modules']:
                    continue
                content_dict[key] = self_dict[key]

            # save as json
            content_dict['class_type'] = 'Container'
            with open(file_path, 'wb') as f:
                pickle.dump(content_dict, f)

        return file_name


# self.container_id = container_id
# self._sub_modules = OrderedDict()
# self._call_time = 0
# self._module
