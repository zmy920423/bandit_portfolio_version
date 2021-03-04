class BanditCollectionData(object):
    def __init__(self, social, N):
        self.social = social
        self.N = N


class BanditData(object):
    def __init__(self, timestamp=-1, bandit_id=-1, arm_reward={}, arm_true_reward={}, arm_context={}, bandit_context=""):
        self.timestamp = timestamp
        self.bandit_id = bandit_id
        self.arm_reward = arm_reward
        self.arm_true_reward = arm_true_reward
        self.arm_context = arm_context
        self.bandit_context = bandit_context

    def __cmp__(self, other):
        return cmp(self.age, other.age)




