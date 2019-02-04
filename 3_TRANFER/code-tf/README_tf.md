Installation de TensorFlow Hub
==============================

Pour le transfer-learning, il est nécessaire d'avoir un modèle déjà entrainé sur une tache proche de celle sur laquelle se porte notre intérêt. Le transfert learning étant très utile en pratique lorsqu'on ne possède soit pas assez de données, soit pas assez de puissance de calcul, la plupart des frameworks de deep-learning proposent des modèles prêts à l'emploi. Ainsi, TensorFlow possède son propre bestiaire de modèles pré-entrainés: TensorTlow Hub. 

Par défaut TensorFlow Hub n'est pas disponible lorsque vous installez tensorflow. L'installation est assez classique.

Avec pip (dans un environnement virtuel ou pas):
``` 
pip install tensorflow-hub
```

Utilisation de TensorFlow Hub 
=============================

La liste des modèles accessibles sur TensorFLow Hub est disponible sur le site https://tfhub.dev. On peut y trouver des modèles entrainés aussi bien sur des taches de classification que de génération (aka GAN...). Aujourd'hui, le sujet étant le transfer-learning notre intérêt se tourne vers les modèles 'Feature Vector' (dans le menu de navigation à gauche du site). En effet, le transfert learning consistant à changer de représentation pour faciliter l'apprentissage notre modèle doit prendre  une image et donner un vecteur représentant cette dernière.

Vous devriez maintenant voir un choix assez important de modèles: inception_v3, mobilenet_v2, inception_resnet_v2, nasnet_large, resnet_v2_152, ... Certains d'entre vous connaissent déjà probablement certains modèles. Si vous n'êtes pas familiers tous ces modèles, je vous conseille chaleureusement d'aller jeter un coup d'oeil aux articles dans lesquels ils ont été introduits afin de gagner en culture. Pour l'instant, nous allons choisir le 'ResNet V2 50', un réseau possédant pas moins de 50 couches et entrainé sur une tache de classification comportant 1000 catégories avec 1.3M d'images (ILSVRC) par Google ! Pour le trouver parmi la multitude de réseaux proposés, utilisez l'outil 'Filter by Network' du menu à gauche de la page. Vous devriez arriver sur la page
https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1.

Sur cette dernière vous deviez trouver des liens vers les articles ayant introduit l'architecture ainsi qu'un exemple d'usage:

```python
import tensorflow-hub as hub

module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1")
height, width = hub.get_expected_image_size(module)
images = ...  # A batch of images with shape [batch_size, height, width, 3].
features = module(images)  # Features with shape [batch_size, num_features].
```

Le module s'obtient facilement avec:

```python
module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1")
```

Le réseau que nous avons choisi prend en entrée des images de taille 224x224. Il est possible d'obtenir cette information avec: 

```python
height, width = hub.get_expected_image_size(module)
```

une fois le module crée, il est possible de l'utiliser comme n'importe quelle fonction:

```python
features = module(images) 
```

ainsi, si `images` est un batch d'images de taille 224x224, features est un batch de feature-vectors de dimension 2048. Attention les images que vous devez donner en entrée du réseau doivent avoir leurs valeures entre 0 et 1 et non entre 0 et 255.

Manque de mémoire ?
-------------------

Si vous rencontrez des problèmes de performance sur votre machine, vous pouvez essayer d'utiliser le module https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2, c'est un réseau beaucoup plus petit qui prend en entrée des images de taille 128x128 et retourne un feature-vector de dimension 1280.

Lecture des données
===================

Maintenant qu'on a notre réseau pré-entraîné, il est temps d'essayer d'adapter la connaissance apprise à une nouvelle tache. Comme on vous l'a surement répété, il faut impérativement séparer les donnés en trois parties:
une partie qui sert à entrainer le modèle (le train set), une partie qui sert à choisir de bons hyper-paramètres (le validation set) et une partie qui sert à estimer la performance finale du modèle. 

Une fois les données séparées, le mieux est d'utiliser l'api `tf.data` (quand on fait du TensorFLow). Il existe de très bons tutoriels pour l'utilisation de cette api sur le site de TensorFlow dont je vous recommande vivement la lecture:

 - https://www.tensorflow.org/guide/datasets 
 - https://www.tensorflow.org/guide/performance/datasets 

Toute la partie lecture des données de l'exemple de code est expliquée dans ces deux tutoriaux.

Transfer learning
=================

Maintenant que nous avons accès à une nouvelle représentation de nos images sous la forme de feature-vectors, on va voir si cela rend le training plus facile (spoil: oui !). La première chose qu'on peut noter, c'est qu'on est passée d'une représentation des images dans un espace vectoriel de dimension 224x224x3 = 150528 à une représentation dans un espace vectoriel de dimension 2048 c'est toujours beaucoup mais c'est nettement mieux.

Pour faire le transfer-learning, nous allons créer un nouveau modèle à partir de notre modèle pré-entrainé. Avec l'api `tf.keras` ça ressemble à ça:

```python
import tensorflow as tf

K = tf.keras

ResNet50 = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1")
height, width = hub.get_expected_image_size(ResNet50)

inputs = K.layers.Input(shape=[height, width, 3])
feature_vector = K.layers.Lambda(ResNet50)(inputs)
outputs = K.layers.Dense(units=3)(feature_vector)

final_model = K.Model(inputs=inputs, outputs=outputs) 
```

C'est presque trop facile !

Fine-Tuning
===========

Pour effectuer le fine-tuning, il faut autoriser la modification des poids de notre ResNet 50 pour cela, il suffit d'ajouter `trainable=True` à la liste des arguments de la fonction `hub.Module(...)`. Cependant lorsque l'on fait du fine-tuning il faut être prudent sur la valeur du learning rate et sur la durée du training car on risque toujours de faire 'oublier' au réseau son expérience passée et d'overfitter sur notre faible ensemble d'apprentissage. Pour le fine-tuning, la documentation conseille d'utiliser le tag `{'train'}` lors de l'instanciation du modèle afin d'activer la batch normalization. Cette dernière étant un peu technique à utiliser avec TensorFlow je vous conseille de ne pas y prêter attention pour le moment.

Manque de mémoire ?
-------------------

Si vous rencontrez des problèmes de performance sur votre machine, vous pouvez essayer d'utiliser le module https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2, c'est un réseau beaucoup plus petit qui prend en entrée des images de taille 128x128 et retourne un feature-vector de dimension 1280. Si vous lisez ce paragraphe alors que tout marchait bien jusqu'à présent, rien de plus normal. Le fine-tuning nécessite plus de mémoire que le transfer learning car il rend nécessaire le calcul du gradient dans le réseau pré-entrainé et donc la sauvegarde en mémoire des activations des couches intermédiaires de ce dernier.
