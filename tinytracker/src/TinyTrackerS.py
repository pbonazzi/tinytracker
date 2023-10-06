import tensorflow as tf
from tensorflow.keras import layers, models

def se_block(inputs, r=16):
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Dense(inputs.shape[-1] // r, activation='relu')(x)
    x = layers.Dense(inputs.shape[-1], activation='sigmoid')(x)
    return layers.Multiply()([inputs, x])

def inverted_residual_block(inputs, filters, expansion, stride):
    x = layers.Conv2D(filters * expansion, 1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    x = layers.DepthwiseConv2D(3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    x = se_block(x)

    x = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if stride == 1 and inputs.shape[-1] == filters:
        x = layers.Add()([x, inputs])

    return x

def lightweight_mobilenetv3(input_shape, num_classes):
    input = layers.Input(shape=input_shape)

    x = layers.Conv2D(8, 3, strides=2, padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    x = inverted_residual_block(x, 8, expansion=1, stride=2)
    x = inverted_residual_block(x, 16, expansion=6, stride=2)
    x = inverted_residual_block(x, 24, expansion=6, stride=1)
    x = inverted_residual_block(x, 32, expansion=6, stride=2)
    x = inverted_residual_block(x, 64, expansion=6, stride=1)

    x = layers.Conv2D(128, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(num_classes, activation='tanh')(x)

    return models.Model(input, output)



class TinyTrackerS(tf.keras.Model):
    def __init__(self, shape = (96,96,3), n_out=2):
        """
        Idea: pretrain full model, discard detection part and then finetune the detection model
        :param shape: image input shape
        :param n_out: 2 for xy and 3 for xyf where f is detecting if there is a face or not
        """
        super(TinyTrackerS, self).__init__()

        self.backbone = self.lightweight_mobilenetv3(shape)

        self.detection = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(n_out, activation='tanh')
        ])

    def lightweight_mobilenetv3(self, input_shape):
        input = layers.Input(shape=input_shape)

        x = layers.Conv2D(8, 3, strides=2, padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.)(x)

        x = inverted_residual_block(x, 8, expansion=1, stride=2)
        x = inverted_residual_block(x, 16, expansion=6, stride=2)
        x = inverted_residual_block(x, 32, expansion=6, stride=1)
        x = inverted_residual_block(x, 32, expansion=6, stride=1)
        x = inverted_residual_block(x, 64, expansion=6, stride=2)

        x = layers.Conv2D(128, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        output = layers.ReLU(6.)(x)
        return models.Model(input, output)

    def call(self, inputs):
        x = self.backbone(inputs)
        x = self.detection(x)
        return x


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, r=16):
        super(SEBlock, self).__init__()
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(input_channels // r, activation='relu')
        self.dense2 = tf.keras.layers.Dense(input_channels, activation='sigmoid')

    @tf.function
    def call(self, inputs):
        x = self.global_pool(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return tf.keras.layers.Multiply()([inputs, x])


class InvertedResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, expansion, stride):
        super(InvertedResidualBlock, self).__init__()
        self.expand = tf.keras.layers.Conv2D(filters * expansion, 1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.ReLU(6.)

        self.dconv = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.ReLU(6.)

        self.squeeze = SEBlock(filters * expansion)
        self.project = tf.keras.layers.Conv2D(filters, 1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.add = tf.keras.layers.Add()
        self.stride = stride

    @tf.function
    def call(self, inputs):
        x = self.expand(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.dconv(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.squeeze(x)

        x = self.project(x)
        x = self.bn3(x)

        if self.stride == 1 and inputs.shape[-1] == x.shape[-1]:
            x = self.add([x, inputs])

        return x
