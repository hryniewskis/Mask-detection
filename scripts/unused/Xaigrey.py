import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
from skimage import color

path = os.getcwd()

model = keras.models.load_model("Conv_nocol.h5")

path = os.path.join(path, 'dataset')

# declare train_path
path_train = os.path.join(path, 'Train')

# declare test_path
path_test = os.path.join(path, 'Test')

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
explainer = lime_image.LimeImageExplainer()

validation_generator = train_datagen.flow_from_directory(path_test,
                                                         color_mode="grayscale",
                                                         batch_size=10,
                                                         target_size=(150, 100))
img = next(validation_generator)[0][0].reshape(150, 100)
plt.imshow(img)


def new_predict_fn(images):
    images = color.rgb2gray(images)
    return model.predict(images.reshape(1, 150, 100, 1))

explanation = explainer.explain_instance(img.astype('double'),
                                         new_predict_fn,
                                         top_labels=2,
                                         hide_color=0,
                                         batch_size=1,
                                         num_samples=1000)




temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)

plt.imshow(mark_boundaries(temp, mask))

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))


temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=30, hide_rest=False, min_weight=0.01)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

model.predict(img.reshape(1, 150, 100, 3))


y_proba = model.predict(img.reshape(1, 150, 100, 3))
y_classes = keras.utils.probas_to_classes(y_proba)

model.predict_classes()

ind =  explanation.top_labels[0]

#Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

#Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()

img_3 = next(train_generator)[0][0]
img_3.shape
color.rgb2gray(img_3.reshape(1, 150, 100)).shape

img_z = next(validation_generator)[0][0].reshape(1, 150, 100, 1)
img_z.shape
model.predict(img_z)