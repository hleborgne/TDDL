import tensorflow_hub as hub

from tensorflow.keras import Model
from  tensorflow.keras.layers import Dense, Softmax

# pour travailler "hors ligne"
# 1 - récupérer le sha1 de l'URL
#     import hashlib
#     handle = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/1"
#     hashlib.sha1(handle.encode("utf8")).hexdigest()
#   ce qui donne '1a3f6d786d1e8f0f1ca9ec98ab08ffc76d7fe55b'
# 2 - récupérer l'archive à partir de :
#    wget https://storage.googleapis.com/tfhub-modules/google/tf2-preview/mobilenet_v2/feature_vector/1.tar.gz
# 3 - préparer le cache et décompresser le fichier
#   mkdir -p /tmp/tfhub/1a3f6d786d1e8f0f1ca9ec98ab08ffc76d7fe55b && cd /tmp/tfhub/1a3f6d786d1e8f0f1ca9ec98ab08ffc76d7fe55b
#   tar xzf /path/to/1.tar.gz
# 4 - ajouter le path du cache dans votre code

# import os
# os.environ["TFHUB_CACHE_DIR"] = '/tmp/tfhub'



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
