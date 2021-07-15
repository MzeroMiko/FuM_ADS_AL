''' Configuration File.
'''

class DATASETS(object):
    def __init__(self):
        self.NUM_TRAIN = 50000
        self.NUM_VAL = 50000 - self.NUM_TRAIN
        self.ROOT = {
            'cifar10': '../cifar10',
            'cifar100': '../cifar100'
        }


class ACTIVE_LEARNING(object):
    def __init__(self):
        # self.TRIALS = 10
        # self.CYCLES = 1 # ????
        self.TRIALS = 3 # Performance = np.zeros((3, 10))
        self.CYCLES = 10 # Performance = np.zeros((3, 10))
        self.ADDENDUM = 1000
        self.SUBSET = 10000


class TRAIN(object):
    def __init__(self):
        self.BATCH = 128
        self.EPOCH = 200
        self.LR = 0.1
        self.MILESTONES = [160]
        self.EPOCHL = 120
        self.MOMENTUM = 0.9
        self.WDECAY = 5e-4
        self.MIN_CLBR = 0.1
        self.MAX_CLBR = 0.1


class CONFIG(object):
    def __init__(self, port=9000):
        self.port = port
        self.DATASET = DATASETS()
        self.ACTIVE_LEARNING = ACTIVE_LEARNING()
        self.TRAIN = TRAIN()


