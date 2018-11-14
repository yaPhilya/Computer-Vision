from keras.preprocessing import image
from keras.applications.xception import preprocess_input
import os
import numpy as np
def train_classifier(gt, img_dir, fast_train=True):
    pass

def classify(model, img_dir):
    ans = {}
    for file in os.listdir(img_dir):
        img = preprocess_input(image.img_to_array(image.load_img('/'.join([img_dir, file]), target_size=(299, 299))))
        res = model.predict(img[np.newaxis])[0].argmax()
        ans[file] = res
    return ans