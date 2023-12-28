import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

class Modelin(Model):
  def __init__(self, input_shape):
    super(Modelin, self).__init__()
    self.prep_flip = layers.RandomFlip()
    self.prep_translation = layers.RandomTranslation(0, 0.2)

  def call(self, data):
    inputs = self.prep_flip(data)
    output = self.prep_translation(inputs)
    return output

class Handle(Model):
  def __init__(self, final_node, activation="softmax"):
    super(Handle, self).__init__()
    self.cnn_one = layers.Conv2D(16, (3, 3), activation="relu", padding="same")
    self.cnn_dua = layers.Conv2D(32, (3, 3), activation="relu", padding="same")
    self.cnn_out = layers.Conv2D(64, (3, 3), activation="relu", padding="same")
    self.pool = layers.MaxPool2D((2, 2))
    self.flat = layers.Flatten()
    self.drop = layers.Dropout(0.2)
    self.dens = layers.Dense(128, activation="relu")
    self.last = layers.Dense(final_node, activation=activation)

  def call(self, datasets):
    pool_one = self.pool(self.cnn_one(datasets))
    pool_dua = self.pool(self.cnn_dua(pool_one))
    pool_out = self.pool(self.cnn_out(pool_dua))

    x = self.flat(pool_out)
    x = self.drop(x)
    x = self.dens(x)
    output = self.last(x)
    return output