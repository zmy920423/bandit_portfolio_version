import torch as torch


class Parameter(torch.Tensor):
    """
    Parameter class, which is a data structure of parameter
    Arguments:
        data(Tensor): Parameter value, which is based on any Tensor Object
    """

    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        return super().__new__(cls, data, *args, **kwargs)

    def clone(self, *args, **kwargs):
        return Parameter(super().clone(*args, **kwargs))

    def to(self, *args, **kwargs):
        new_obj = Parameter([])
        new_obj.data = super().to(*args, **kwargs)

        return new_obj


if __name__ == '__main__':
    a = Parameter(torch.eye(6))
    b = a + a
    a = b
    print(type(a), type(b))

    a = torch.nn.Parameter(torch.eye(6))
    b = a * 1
    a = b
    print(type(a), type(b))
