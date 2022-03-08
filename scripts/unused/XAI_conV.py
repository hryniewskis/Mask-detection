import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

path = os.getcwd()

model = keras.models.load_model("scripts/Conv.h5")

path = os.path.join(path, 'dataset')

# będziemy przeglądać zdjęcia ze zbioru testowego
path_test = os.path.join(path, 'test')

datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = datagen.flow_from_directory(path_test,
                                                   batch_size=1,
                                                   target_size=(150, 100))


img = next(validation_generator)[0].reshape(150, 100, 3)
plt.imshow(img)

explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(img.astype('double'),
                                         model.predict,
                                         top_labels=2,
                                         hide_color=0,
                                         num_samples=1000)

# Czemu zaznaczył to co zaznaczył?
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp, mask))

# A co w sumie zaznaczył?
print(model.predict(img.reshape(1,150, 100,3)))

# 10 obszarów o największym wpływie na to czy uzna, że ma maskę (zielone) czy nie (czerwone)
temp, mask = explanation.get_image_and_mask(label=0, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

# obszary o wpływie co najmniej 0.2
temp, mask = explanation.get_image_and_mask(label=0, positive_only=False, hide_rest=False, min_weight=0.2)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))


# Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[0])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

# Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()
