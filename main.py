
from autokerasens.image import AutokerasImageEnsemble


import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.datasets import cifar10
import numpy as np
import gc
from keras import backend as K 


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    y_train = to_categorical(y_train, num_classes=10)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_data()


ak_ensemble = AutokerasImageEnsemble(project_name = "cifar10",
                                     max_trials=1,
                                     directory="/content/drive/My Drive/cifar10/output_cifar10",
                                     objective="val_loss",
                                     overwrite=False)


ak_ensemble.fit(x_train, y_train, epochs=100)

K.clear_session()
gc.collect()