import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.image as mpimg
import numpy as np
import cv2


class StampClassifierSynthetic:

    def train(self):
        batch_size = 128
        epochs = 20
        IMG_HEIGHT = 100
        IMG_WIDTH = 100

        train_dir = "training_images"
        train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data

        train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                                   directory=train_dir,
                                                                   shuffle=True,
                                                                   target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                   class_mode='sparse')

        label_map = (train_data_gen.class_indices)
        print(label_map)

        model = models.Sequential()
        model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(11))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit_generator(
            train_data_gen,
            steps_per_epoch=26928 // batch_size,
            epochs=epochs,
        )

        model.save("model.h5")


    def predict(self, folder_to_classify):
        model = models.load_model('./model.h5')
        labels = {'10': 0, '15': 1, '150': 2, '20': 3, '25': 4, '250': 5, '30': 6, '5': 7, '50': 8, '75': 9, '80': 10}
        label_map = {v: k for k, v in labels.items()}
        all_imgs = []
        img_order = []
        for i, filename in enumerate(os.listdir(folder_to_classify)):
            img_order.append(filename)
            img = mpimg.imread(os.path.join(folder_to_classify, filename))
            img = cv2.resize(np.int16(img), (100,100))
            img = np.array(img, dtype="int32")
            img = np.expand_dims(img, 0)
            img = img * (1. / 255)
            if i == 0:
                all_imgs = img
            else:
                all_imgs = np.concatenate((all_imgs, img))
        predictions = model.predict(all_imgs)
        prediction_labels = [label_map[np.argsort(prediction)[-1]] for prediction in predictions]
        return prediction_labels, img_order
