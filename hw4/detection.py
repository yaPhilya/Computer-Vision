import os
import numpy as np
from skimage import io
from skimage.transform import resize
import keras
import keras.layers as L

def train_detector(gt, img_dir, fast_train=True):
    if fast_train:
        points = []
        files = []
        it = 256
        for name, ans in gt.items():
            files.append(name)
            points.append(ans)
            it -= 1
            if it == 0:
                break
        points = np.array(points, dtype=np.float64)
        images = []
        shapes = []
        for file in files:
            im = io.imread('/'.join([img_dir, file]))
            images.append(resize(im, (100, 100, 3)))
            shapes.append((im.shape[0], im.shape[1]))
        images = np.array(images)
        shapes = np.array(shapes)
        points[:, 0::2] = points[:, 0::2] / shapes[:, 1, np.newaxis] * 100.0
        points[:, 1::2] = points[:, 1::2] / shapes[:, 0, np.newaxis] * 100.0
        model = keras.models.Sequential()
        model.add(L.Convolution2D(filters=16, kernel_size=3, input_shape=(100, 100, 3)))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))
        model.add(L.Convolution2D(filters=32, kernel_size=3))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))
        model.add(L.MaxPooling2D())

        model.add(L.Convolution2D(filters=64, kernel_size=3))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))
        model.add(L.Convolution2D(filters=128, kernel_size=3))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))
        model.add(L.MaxPooling2D())

        model.add(L.Convolution2D(filters=256, kernel_size=3))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))
        model.add(L.Convolution2D(filters=512, kernel_size=3))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))
        model.add(L.MaxPooling2D())

        model.add(L.Flatten())
        model.add(L.Dense(units=128, activation='relu'))
        model.add(L.Dense(units=64, activation='relu'))
        model.add(L.Dense(units=28))
        model.compile('adam', loss='mse')
        model.fit(images, points, batch_size=128, epochs=1, verbose=0)
        
    
def detect(model, test_img_dir):
    ans = {}
    for file in os.listdir(test_img_dir):
        orig_img = io.imread('/'.join([test_img_dir,file]))
        img = resize(orig_img, (100, 100, 3))
        res = model.predict(img[np.newaxis])[0]
        res[0::2] = res[0::2] / 100 * orig_img.shape[1]
        res[1::2] = res[1::2] / 100 * orig_img.shape[0]
        ans[file] = res
    return ans
        