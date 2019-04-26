import tensorflow as tf
from tensorflow.keras.applications import Xception, VGG19, ResNet50, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D


class XceptionBottleneck(tf.keras.Model):

    def __init__(self):
        super(XceptionBottleneck, self).__init__()

        self.initial_layers = Xception(weights='imagenet', include_top=False)
        self.pooling_layer = GlobalAveragePooling2D()

    def call(self, inputs):
        result = self.initial_layers(inputs)
        result = self.pooling_layer(result)
        return result


class VGG19Bottleneck(tf.keras.Model):

    def __init__(self):
        super(VGG19Bottleneck, self).__init__()

        self.initial_layers = VGG19(weights='imagenet', include_top=False)
        self.pooling_layer = GlobalAveragePooling2D()

    def call(self, inputs):
        result = self.initial_layers(inputs)
        result = self.pooling_layer(result)
        return result


class ResNet50Bottleneck(tf.keras.Model):

    def __init__(self):
        super(ResNet50Bottleneck, self).__init__()

        self.initial_layers = ResNet50(weights='imagenet', include_top=False)
        self.pooling_layer = GlobalAveragePooling2D()

    def call(self, inputs):
        result = self.initial_layers(inputs)
        result = self.pooling_layer(result)
        return result


class DenseNet121Bottleneck(tf.keras.Model):

    def __init__(self):
        super(DenseNet121Bottleneck, self).__init__()

        self.initial_layers = DenseNet121(weights='imagenet', include_top=False)
        self.pooling_layer = GlobalAveragePooling2D()

    def call(self, inputs):
        result = self.initial_layers(inputs)
        result = self.pooling_layer(result)
        return result


class OneLayerClassifier(tf.keras.Model):

    def __init__(self, n_classes):
        super(OneLayerClassifier, self).__init__()
        # Define the layer(s) you would like to use for your classifier
        self.dense_layer = tf.keras.layers.Dense(units=n_classes)

    def call(self, inputs):
        # Set this up appropriately, will depend on how many layers you choose
        result = self.dense_layer(inputs)

        return result
