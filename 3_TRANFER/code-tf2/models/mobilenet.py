import tensorflow_hub as hub

from  tensorflow.keras import (Model, Dense, Softmax)



# Define our model with keras model subclassing
class MobileNet(K.Model):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.backbone = hub.KerasLayer(
            'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/1',
            trainable=FLAGS.fine_tune,
            output_shape=[1280],
        )
        self.dense = Dense(units=3)
        self.softmax = Softmax()
    
    def call(self, x):
        h = self.backbone(x)
        h = self.dense(h)
        y = self.softmax(h)
        return y
