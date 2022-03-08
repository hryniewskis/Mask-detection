import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import os

# model 1
# link to source: https://github.com/chandrikadeb7/Face-Mask-Detection

trained_model = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

for layer in trained_model.layers:
    layer.trainable = False

last_layer = trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


path = os.getcwd()
path = os.path.join(path, 'dataset')

# declare train_path
path_train = os.path.join(path, 'Train')

# declare test_path
path_test = os.path.join(path, 'Test')



train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0/255.)

train_generator = train_datagen.flow_from_directory(path_train,
                                                    batch_size=64,
                                                    class_mode='binary',
                                                    target_size=(224, 224))

validation_generator = test_datagen.flow_from_directory(path_test,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(224, 224))

model.fit(train_generator,
          validation_data=validation_generator,
          steps_per_epoch=12,
          epochs=5,
          validation_steps=9,
          verbose=1)

model.predict(next(train_generator)[0])


model.save("imagenet.h5")

