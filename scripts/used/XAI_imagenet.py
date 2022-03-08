import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np


path = os.getcwd()

model= keras.models.load_model("scripts/imagenet.h5")

path = os.path.join(path, 'dataset')

# We'll be looking at pictures in the test set
path_test = os.path.join(path, 'test')

datagen = ImageDataGenerator(rescale=1.0 / 255)

generator = datagen.flow_from_directory(path_test,
                                        batch_size=1,
                                        target_size=(224, 224))


img = next(generator)[0].reshape(224, 224, 3)

#a peek at the image of which explanation we'll be looking at
plt.imshow(img)

explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(img.astype('double'),
                                         model.predict,
                                         top_labels=2,
                                         hide_color=0,
                                         num_samples=1000)

# Why model has classified the image in this way?
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp, mask))

# What is the probability that the person has a mask on the image?
print(model.predict(img.reshape(1,224,224,3)))

# 10 regions with the highest weight on the result (mask- red, no mask- green)
temp, mask = explanation.get_image_and_mask(label=0, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

# regions with the weight at least 0.1
temp, mask = explanation.get_image_and_mask(label=0, positive_only=False, num_features=10, hide_rest=False, min_weight=0.2)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

#Mapping each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[0])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

#Ploting of the visualization, makes more sense if a symmetrical colorbar is used
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()
