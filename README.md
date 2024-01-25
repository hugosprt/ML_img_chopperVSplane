# Projet de Classification Avion vs Hélicoptère
Projet realise grace a la video de defend intelligence sur le CNN.

Ce projet consiste en la création d'un modèle de deep learning pour classer des images en deux catégories : avions et hélicoptères. Le modèle est construit à l'aide de TensorFlow et Keras, et il est capable de prédire si une image donnée représente un avion ou un hélicoptère avec une certaine confiance.

## Contenu

- [Prérequis](#prérequis)
- [Utilisation](#utilisation)
- [Exemple](#exemple)
- [Auteur](#auteur)


## Prérequis

Avant de commencer, assurez-vous d'avoir installé les bibliothèques Python suivantes :
- TensorFlow
- NumPy
- Matplotlib

Vous pouvez les installer en utilisant `pip` :

```bash
pip install tensorflow numpy matplotlib
Installation


Pour classer une image comme avion ou hélicoptère, utilisez la commande suivante :
bash
Copy code
python cnn.py chemin_vers_votre_image.jpg
Assurez-vous de remplacer chemin_vers_votre_image.jpg par le chemin absolu de l'image que vous souhaitez classifier.

Exemple

bash
Copy code
python cnn.py chemin_vers_votre_image.jpg
Le modèle effectuera une prédiction et affichera la catégorie prédite ainsi que la confiance dans cette prédiction.

Auteur

Hugo sprt
