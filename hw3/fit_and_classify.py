import numpy as np
from skimage.filters import sobel_h, sobel_v
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

def get_grad(image):
    def get_luminance(image):
        RC = 0.299
        GC = 0.587
        BC = 0.114
        f = image.astype(np.float64)
        return f[:,:,0] * RC + f[:,:,1] * GC + f[:,:,2] * BC
    luminance = get_luminance(image)
    s_x = -sobel_v(luminance)
    s_y = sobel_h(luminance)
    grad = np.sqrt(s_x ** 2 + s_y ** 2)
    angle = np.arctan2(s_y, s_x)
    return grad, angle

def get_cells(grad, angle, orient=9, pixels_per_cell=(8, 8), step=(8, 8)):
    rows_per_cell = pixels_per_cell[0]
    columns_per_cell = pixels_per_cell[1]
    shape = grad.shape
    cells = np.zeros((int(np.ceil(shape[0] / step[0])), int(np.ceil(shape[1] / step[1])), orient))
    for i in range(0, shape[0], step[0]):
        for j in range(0, shape[1], step[1]):
            i_2 = i + rows_per_cell
            j_2 = j + columns_per_cell
            hist, _ = np.histogram(angle[i:i_2, j:j_2], bins=orient, range=(-np.pi, np.pi), weights=grad[i:i_2, j:j_2])
            cells[i // step[0], j // step[1]] = hist
    return cells

def get_blocks(cells, cells_per_block=(2, 2), step=(1, 1), eps=0.00001):
    rows_per_block = cells_per_block[0]
    columns_per_block = cells_per_block[1]
    shape = cells.shape
    blocks = np.zeros((int(np.ceil(shape[0] / step[0])), int(np.ceil(shape[1] / step[1])), 
                       shape[2] * rows_per_block * columns_per_block))
    for i in range(0, shape[0], step[0]):
        for j in range(0, shape[1], step[1]):
            i_2 = i + rows_per_block
            j_2 = j + columns_per_block
            block = np.zeros(shape[2] * rows_per_block * columns_per_block)
            block[:np.ravel(cells[i:i_2, j:j_2, :]).shape[0]] = np.ravel(cells[i:i_2, j:j_2, :])
            blocks[i // step[0], j // step[1]] = block / np.sqrt(np.sum(block ** 2) + eps)
    return blocks

def extract_hog(image):
    image = resize(image, output_shape=(64, 64))
    grad, angle = get_grad(image)
    cells = get_cells(grad, angle, orient=8, pixels_per_cell=(8,8), step=(8,8))
    blocks = get_blocks(cells, cells_per_block=(2, 2), step=(1, 1))
    return blocks.flatten()

def fit_and_classify(train_features, train_labels, test_features):
    model = LinearSVC()
    model.fit(train_features, train_labels)
    return model.predict(test_features)