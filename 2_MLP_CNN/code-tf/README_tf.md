I - Lecture les données
=======================

Pour lire des données TensorFlow je vous recommande d'utiliser l'api `tf.data`. De très bons tutoriaux existent sur le site de TensorFlow:

 - https://www.tensorflow.org/guide/datasets 
 - https://www.tensorflow.org/guide/performance/datasets 

Dans les grandes lignes, cette api utilise deux classes `tf.data.Dataset` et `tf.data.Iterator`. Comme leurs noms l'indiquent `tf.data.Dataset` représente un dataset et `tf.data.Iterator` permet d'itérer sur ce dernier. L'intérêt de cette api c'est qu'elle nous donne accès à de nombreuses fonctions bien utiles pour créer des batchs, mélanger nos datasets, augmenter nos données en temps réel ou encore optimiser le traitement des données. On va donc se servir de ces deux objets pour créer notre pipeline d'objets.

Recuperer MNIST
---------------

MNIST (http://yann.lecun.com/exdb/mnist/) est une base de données d'images en niveau de gris de chiffres manuscrits (de taille 28x28). elle contient 60 000 images de train et de 10 000 images de test. C'est une base de données très utilisée dans le domaine de l'IA surtout dans l'enseignement et dans la recherche (très) expérimentale.

Il est possible de récupérer les données directement dans TensorFlow en une ligne de code:

```python
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
```

Les données ainsi obtenues, elles sont sous forme de `numpy.ndarray`s. Nous allons donc les convertir en `tf.data.Dataset`s puis nous allons créer des `tf.data.Iterator`s.

Créer un `tf.data.Dataset`
--------------------------

La première chose à faire est de bien sûr diviser notre base de données d'apprentissage en deux parties: une pour l'apprentissage des poids (le train-set) et une pour l'apprentissage (manuel) des hyper-parametres (le validation-set). Une fois que cela est fait, on crée trois datasets de la façon suivante:

```python
dataset = tf.data.Dataset.from_tensor_slices((x, y)) 
```

Utiliser l'api `tf.data` 
------------------------

Puis s'il s'agit du train set, on le mélange et on le 'répète' afin de pouvoir itérer plusieurs fois dessus:

```python
dataset = dataset.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=60000))
```

NB: les plus curieux d'entre vous se demanderont probablement ce que signifie `buffer_size=100000`. C'est la taille du buffer utilisé pour mélanger le dataset.
L'algorithme est le suivant: à chaque fois qu'on tire un nouvel élément dans le dataset, on en tire un au hasard dans les `buffer_size` éléments suivants. il faut donc choisir un buffer_size de la taille du dataset pour que le mélange soit uniforme. 

Souvent il est nécessaire de prétraiter les données et de les regrouper en batchs. Avec l'api `tf.data` cela donne: 

```python
def parse(image, label):
    """function that normalizes the examples"""
    image = tf.reshape(image, [28,28,1]) # add channel dimension
    image = tf.cast(image, tf.float32)
    image = image / 127.5 - 1.0

    label = tf.cast(label, tf.int32)

    return {'image': image, 'label': label}

dataset = dataset.apply(tf.data.experimental.map_and_batch(
    map_func=parse, batch_size=batch_size, num_parallel_batches=8))
        
```

Il ne nous reste alors qu'à créer notre `tf.data.Iterator`:

```python
iterator = dataset.make_one_shot_iterator()
```

II - Construire un modèle
=========================

Pour construire un modèle avec TensorFlow, le plus pratique est d'utiliser l'api 'haut niveau' de TensorFlow: `tf.keras`. Bien sûr il est possible de recoder toutes les couches d'un réseau mais c'est souvent une perte de temps. Peut-être serez-vous amenés à coder vos propres couches expérimentales un jour, mais là encore, l'api `tf.keras` permet la creation de nouveaux types de couches https://keras.io/layers/writing-your-own-keras-layers/.

Il existe plusieurs façon de créer un modèle avec `tf.keras` en créant une classe fille de `tf.keras.Model` ou en utilisant l'api fonctionnelle de création de modèle. Nous allons détailler cette dernière car elle me paraît plus intuitive. il existe un tutoriel pour créer des modèles avec `tf.keras` sur le site de TensorFLow: https://www.tensorflow.org/api_docs/python/tf/keras/models/Model. Ici je vais tenter d'expliquer brièvement l'essentiel.


Creation d'un modèle avec une approche fonctionnelle
----------------------------------------------------

Rien de mieux qu'un example pour comprendre:

```python
x = tf.keras.layers.Input(shape=[28,28,1], dtype=tf.float32)
h = tf.keras.layers.Conv2D(64, 5, padding='valid', activation=tf.nn.relu)(x)
h = tf.keras.layers.MaxPool2D()(h) 
h = tf.keras.layers.Conv2D(64, 5, padding='valid', activation=tf.nn.relu)(h) 
h = tf.keras.layers.MaxPooling2D()(h)
h = tf.keras.layers.Flatten()(h)
h = tf.keras.layers.Dropout(0.5)(h)
h = tf.keras.layers.Dense(units=2*ch)(h)
y = tf.keras.layers.Dense(units=10)(h)

model = tf.keras.Model(inputs=x, outputs=y)
```

La seule contrainte, est lq suivante: entre l'entrée et la sortie du modèle il ne doit y avoir QUE des `tf.keras.layers.Layer` (comme dans l'exemple). Cependant, vous aurez parfois besoin d'une opération qui ne s'exprime pas en 'layer', heureusement keras met à disposition une layer bien pratique: `tf.keras.layers.Lambda` dont l'utilisation est assez explicite:

/```python
x = tf.keras.layers.Lambda(ma_fonction)(x)
```

Il est aussi possible de créer ses propres layers en créant une classe fille de `tf.keras.layers.Layer`: https://www.tensorflow.org/tutorials/eager/custom_layers


III - Apprendre les poids du modèle
===================================

Pour entrainer notre modèle il va nous falloir définir une loss et un optimiseur;

```python
training = tf.placeholder_with_default(False, [])
logits = model(images, training=training)

loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels, 10), logits)
optimizer = tf.train.AdamOptimizer()

train_op = optimizer.minimize(loss)
```

NB: la variable `training` sert à indiquer au modèle s'il est en phase d'apprentissage ou non (s'il faut utiliser ou non le dropout, comment doit se comporter la batch normalisation, etc...)

Une fois ces opérations définies, il ne reste plus qu'à entrainer le réseau. Pour cela on crée une `tf.Session`, on initialise les variables et on effectue la boucle de train:

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    for _ in range(steps):
        sess.run(train_op, {training: True})

```

Et voilà ! Pour suivre l'avancement du training et pour évaluer le modèle référez-vous au code complet que l'on vous a fourni.