import tensorflow_hub as hub

from tensorflow.keras import Model
from  tensorflow.keras.layers import Dense, Softmax



# Define our model with keras model subclassing
class MobileNet(Model):
    def __init__(self, fine_tune=False):
        super(MobileNet, self).__init__()
        self.backbone = hub.KerasLayer(
            'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/1',
            trainable=fine_tune,
            #output_shape=[1280],
        )
        self.dense = Dense(units=6)
        self.softmax = Softmax()
    
    def call(self, x):
        h = self.backbone(x)
        h = self.dense(h)
        y = self.softmax(h)
        return y
