import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import argparse 

# Chemin vers les dossiers d'entraînement
base_dir = './'  
train_dir = os.path.join(base_dir, 'train')
train_plane_dir = os.path.join(train_dir, 'plane')
train_chopper_dir = os.path.join(train_dir, 'chopper')

# Paramètres de prétraitement et d'entraînement
batch_size = 32
img_height = 360
img_width = 360

# Préparation des données
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

# Construction du modèle
model = models.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))
])

# Compilation du modèle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entraînement du modèle
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Fonction pour prédire une nouvelle image
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "Cette image appartient à {} avec une confiance de {:.2f}%"
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

def main(image_path):
    predict_image(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classifiez une image en tant que plane ou chopper.')
    parser.add_argument('image_path', type=str, help='Le chemin complet de l\'image à classer.')
    args = parser.parse_args()

    main(args.image_path)