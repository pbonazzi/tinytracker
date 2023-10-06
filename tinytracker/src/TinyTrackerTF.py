import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV3Small

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import Sequential

class ItrackerImageModel(Model):
    # Used for both eyes (with shared weights) and the face (with unique weights)
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = Sequential([
            Conv2D(96, kernel_size=(7, 7), strides=(4, 4), padding='valid'),
            ReLU(),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            BatchNormalization(),
            Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same', groups=2),
            ReLU(),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            BatchNormalization(),
            Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            ReLU(),
            Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='valid'),
            ReLU()
        ])

    def call(self, inputs):
        return self.features(inputs)

def create_grid(g, sx, sy):

    linx = tf.linspace(g[0],g[2],sx)
    liny = tf.linspace(g[1],g[3],sy)
    return tf.meshgrid(linx, liny)
class bakedModel(Model):
    def __init__(self, model, shape):
        super(bakedModel, self).__init__()
        self.model = model
        gridX,gridY = create_grid([-0.1,-0.1,0.1,0.1],shape[0],shape[1])
        gridX = tf.expand_dims(gridX, axis=-1)
        gridY = tf.expand_dims(gridY, axis=-1)
        self.grid  = tf.expand_dims(tf.concat([gridX, gridY], axis=-1),0)

    def call(self, inputs):

        return self.model(tf.concat([inputs, self.grid], axis=-1))

class TinyTracker(Model):
    def __init__(self, in_channels=3, backbone = "mobilenetv3"):

        super(TinyTracker, self).__init__()
        # we use the smallest mobile net (minimalistic=True)
        # we remove preprocessing, is done outside

        self.conv_g = layers.Conv2D(in_channels, kernel_size=1, strides=1, padding='valid')
        if backbone == "mobilenetv3":
            self.faceModel = MobileNetV3Small(weights='imagenet', include_top=False, minimalistic=True, include_preprocessing=False)
        else:
            self.faceModel = ItrackerImageModel()
        self.conv_fm = layers.Conv2D(2, kernel_size=1, strides=1, padding='valid')
        n_hidden = 128

        # Joining everything
        self.fc = tf.keras.Sequential([
            layers.Dense(n_hidden, activation='relu'),
            layers.Dense(2, activation='tanh')
        ])



    #@tf.function
    def call(self, faces):


        faces = self.conv_g(faces)
        # Face net
        xFace = self.faceModel(faces)
        xFace = self.conv_fm(xFace)
        xFace = tf.reshape(xFace, (xFace.shape[0], -1))  # equivalent to Flatten in PyTorch

        x = self.fc(xFace)
        return x