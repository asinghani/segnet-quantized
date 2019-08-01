import os
import yaml

# General
random_seed = 42

class Config(object):
    def __init__(self):
        # Dataset Params
        self.train_test_split = 0.1
        self.augment_ratio = 0.5

        self.image_size = (224, 224)

        self.input_shape = (self.image_size[0], self.image_size[1], 3) # should height and width be reversed ??

        # Hyperparams
        self.samples_per_epoch = 200
        self.total_epoch = 1000
        self.batch_size = 16
        self.test_batch_size = 16

        self.learning_rate = 3.5e-4
        self.dropout = 0.4
        self.l2_constant = 1e-3

        self.mobilenet_alpha = 1.0
        self.include_softmax = True

        self.init_weights = None #"/home/anish/mobilenet_weights.h5" # path to weights

    def serialize(self):
        return yaml.dump(self.__dict__)
