import torch
from collections import OrderedDict
from ..parameter import Parameter
import datetime
import pickle


class Module(object):
    """
    Base class for all parameter modules, which is parameter space, without bandit structure.
    """
    def __init__(self, module_id):
        """
        Initializes internal Module state.
        """
        self.__id = module_id
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._decide_hooks = OrderedDict()
        self._call_time = 0

    def set_id(self, id):
        self.__id = id

    def get_id(self):
        return self.__id

    def get_time(self):
        return self._call_time

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self._call_time = datetime.datetime.now()
        self.update(*args, **kwargs)
        self.register_decide_hooks(self._decide_hooks.keys())
        self._call_time = datetime.datetime.now() - self._call_time

    def decide(self, *args, **kwargs):
        return self._decide_hooks

    def parameters(self):
        for name, param in self.named_parameters():
            yield param

    def named_parameters(self):
        memo = set()
        for name, param in self._parameters.items():
            if param is not None and param not in memo:
                memo.add(param)
                yield name, param

    def buffers(self):
        """Returns an iterator over module buffers.
        Yields:
            bandit.Parameter: buffer
        """
        for name, buffer in self.named_buffers():
            yield buffer

    def named_buffers(self):
        memo = set()
        for name, buffer in self._buffers.items():
            if buffer is not None and buffer not in memo:
                memo.add(buffer)
                yield name, buffer

    def register_buffer(self, name, buffer):
        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("buffer name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))

        if buffer is None:
            self._buffers[name] = None
        elif not isinstance(buffer, torch.Tensor):
            raise TypeError("cannot assign '{}' object to torch.Tensor '{}' "
                            "(torch.Tensor or None required)"
                            .format(type(buffer), name))
        else:
            self._buffers[name] = buffer

    def register_parameter(self, name, param):
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, str):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter) and not isinstance(param, torch.Tensor):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(bandit.Parameter or None required)"
                            .format(type(param), name))
        else:
            self._parameters[name] = param.clone().detach()

    def register_decide_hooks(self, name_list):
        if '_decide_hooks' not in self.__dict__:
            raise AttributeError(
                "cannot assign decide hook before Module.__init__() call")
        else:
            for name in name_list:
                if name in self._parameters.keys():
                    self._decide_hooks[name] = self._parameters[name]
                elif name in self._buffers.keys():
                    self._decide_hooks[name] = self._buffers[name]
                elif name in self.__dict__.keys():
                    self._decide_hooks[name] = self.__dict__[name]
                else:
                    raise KeyError("attribute '{}' is not in parameters or buffers".format(name))

    def apply(self, fn):
        fn(self)
        return self

    def _apply(self, fn):
        for key, para in self._parameters.items():
            if para is not None:
                self._parameters[key] = fn(para)
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        return self

    def cuda(self, device=None):
        """Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cuda(device))

    def cpu(self):
        """Moves all model parameters and buffers to the CPU.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cpu())

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if isinstance(value, torch.Tensor):
                remove_from(self.__dict__, self._buffers)
                self.register_parameter(name, value)
            elif value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(Parameter or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            buffers = self.__dict__.get('_buffers')
            if isinstance(value, torch.Tensor):
                if buffers is None:
                    raise AttributeError(
                        "cannot assign buffers before Module.__init__() call")
                remove_from(self.__dict__, self._parameters)
                self.register_buffer(name, value)
            elif buffers is not None and name in buffers:
                if value is not None:
                    raise TypeError("cannot assign '{}' as buffer '{}' "
                                    "(torch.Tensor or None expected)"
                                    .format(torch.typename(value), name))
                self.register_buffer(name, value)
            else:
                object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        else:
            object.__delattr__(self, name)

    def save(self, file_folder_path, prev_path=''):
        if '_parameters' not in self.__dict__.keys():
            raise AttributeError(
                "cannot assign parameter before {}.__init__() call".format(type(self).__name__))

        content_dict = {}
        if prev_path:
            raise NotImplementedError

        else:
            self_dict = self.__dict__
            file_name = type(self).__name__ + '@' + \
                        str(datetime.datetime.now()).replace('-', '_').replace(' ', '_')
            file_path = file_folder_path + '/' + file_name

            # save parameters
            for key in self_dict.keys():
                content_dict[key] = self_dict[key]

            # save as json
            content_dict['class_type'] = 'Module'
            with open(file_path, 'wb') as f:
                pickle.dump(content_dict, f)

        return file_name
