from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

#linkg: https://data-flair.training/blogs/face-mask-detection-with-python/

# model 2
model = Sequential([
    Conv2D(100, (3, 3), activation='relu', input_shape=(150, 100, 1)),
    MaxPooling2D(2, 2),

    Conv2D(100, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])



path=os.getcwd()
path=os.path.join(path,'dataset')

# declare train_path
path_train=os.path.join(path,'Train')

# declare test_path
path_test=os.path.join(path,'Test')
train_datagen=ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(path_train,
                                                    batch_size=10,
                                                    color_mode="grayscale",
                                                    target_size=(150, 100))

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = validation_datagen.flow_from_directory(path_test,
                                                              batch_size=10,
                                                              color_mode="grayscale",
                                                              target_size=(150, 100))

history = model.fit_generator(train_generator,
                              epochs=6,
                              validation_data=validation_generator)


img = next(train_generator)[0][0].reshape(1, 150, 100)
model.predict(img)
model.predict(next(train_generator)[0])
model.predict(next(train_generator)[0])

model.save("Conv_nocol.h5")

