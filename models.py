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


class FeedForwardClassifier(tf.keras.Model):

    def __init__(self, n_classes, layer_dims, activations, drop_out_flag=False, drop_out_rate=0.5):
        # layer_dims is number of layer before the last layer
        super(FeedForwardClassifier, self).__init__()
        self.dense_layers = []
        for idx, layer_dim in enumerate(layer_dims):
            self.dense_layers.append(tf.keras.layers.Dense(units=layer_dim, activation=activations[idx]))
            if drop_out_flag:
                self.dense_layers.append(tf.keras.layers.Dropout(drop_out_rate))
        self.final_layer = tf.keras.layers.Dense(units=n_classes)

    def call(self, inputs):
        result = tf.identity(inputs)
        for i in range(len(self.dense_layers)):
            result = self.dense_layers[i](result)
        result = self.final_layer(result)
        return result
