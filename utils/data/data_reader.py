import json
from utils.data.load_constructor import default_load_constructor, tag_load_constructor


class DataReader(object):
    def __init__(self, file_path, data2bandit_data=default_load_constructor):
        self.fptr = open(file_path, mode='r')
        self.data2bandit_data = data2bandit_data

    def __del__(self):
        self.fptr.close()

    def json2obj(self, json_obj):
        x = json.loads(json_obj)
        python_obj = self.data2bandit_data(x)
        return python_obj

    def fetch_all(self):
        tmp = self.fptr.tell()
        self.fptr.seek(0, 0)
        lines = self.fptr.readlines()
        res = []
        for l in lines:
            res.append(self.json2obj(l))
        self.fptr.seek(tmp, 0)
        return res

    def fetch_one(self):
        line = self.fptr.readline()
        if not line:
            return None
        return self.json2obj(line)

    def fetch_next_batch(self, batch_size=1):
        res = []
        for i in range(batch_size):
            l = self.fptr.readline()
            if not l:
                break
            res.append(self.json2obj(l))
        if len(res) == 0:
            return None
        return res

    def seek_fpt(self, offset):
        self.fptr.seek(offset, 0)
