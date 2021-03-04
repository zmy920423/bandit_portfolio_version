import torch as torch
from utils.data import BanditData


def dict2tensor(dicts):
    if dicts:
        for key, value in dicts.items():
            dicts[key] = torch.tensor(value)
        return dicts
    else:
        return None


def default_load_constructor(json_object):
    if json_object["bandit_context"] != "":
        bandit_context = torch.tensor(json_object["bandit_context"])
    else:
        bandit_context = None
    return BanditData(timestamp=json_object["timestamp"], bandit_id=json_object["bandit_id"],
                      arm_reward=dict2tensor(json_object["arm_reward"]),
                      arm_true_reward=dict2tensor(json_object["arm_true_reward"]),
                      arm_context=dict2tensor(json_object["arm_context"]), bandit_context=bandit_context)

def tag_load_constructor(json_object):
    bandit_context = {'context':{},'tagging':{}, 'candidate_tags':{}, 'full_reviews':{}}
    bandit_context['context'] = torch.tensor(json_object["bandit_context"]['context'])
    bandit_context['tagging'] = json_object["bandit_context"]['tagging'] # dict = {arm: {tag:reward}}：对哪个arm打了的tags，及其reward是多少
    bandit_context['candidate_tags'] = json_object["bandit_context"]['candidate_tags'] # tag做可解释的候选集
    bandit_context['full_reviews'] = json_object["bandit_context"]['full_reviews'] # 专门给Amazon数据集留的

    return BanditData(timestamp=json_object["timestamp"], bandit_id=json_object["bandit_id"],
                      arm_reward=dict2tensor(json_object["arm_reward"]),
                      arm_true_reward=dict2tensor(json_object["arm_true_reward"]),
                      arm_context=dict2tensor(json_object["arm_context"]), bandit_context=bandit_context)
