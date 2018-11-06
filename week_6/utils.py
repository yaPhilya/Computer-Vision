import glob
import os

import tensorflow as tf


def _parse_function(example_proto, i_size):
    """Parses serialized dataset object and decodes images into an elemnt of tf.Dataset.

        Args:
            example_proto (tf.Tensor): an element of the tfrecord (dataset element serialized to a scalar string Tensor)
            i_size (int32):  desired size of the input image to the network
        Returns:
            dataset_element (dict): an element of the dataset {'img': 3D Tensor, 'mask': 3D Tensor,
                'height': int32, 'width': int32, 'filename': str, 'image_size': int32}
    """
    # decoding scheme
    keys_to_features = {
        'image_raw': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'filename': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'height': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'width': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'mask_raw': tf.FixedLenFeature(
            (), tf.string, default_value=''),
    }
    # decoding of serialized element
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    # images decoding
    img = tf.image.decode_jpeg(parsed_features['image_raw'], channels=3)
    mask = tf.squeeze(tf.image.decode_png(parsed_features['mask_raw'], channels=1), axis=-1)

    return {'img': img, 'mask': mask, 'height': parsed_features['height'], 'image_size': i_size,
            'width': parsed_features['width'], 'filename': parsed_features['filename']}


def construct_dataset(path_to_tfrecords, split, n_jobs=6, i_size=224):
    """Construct tensorflow dataset from tfrecord files.

        Args:
            path_to_tfrecords (str): path to a folder with tfrecords (name pattern is train-0000-of-0004.tfrecord)
            split (str): type of data split ('train' or 'val')
            n_jobs (int32, optional): number of parallel jobs to run
            i_size (int32, optional): desired size of the input image to the network
        Returns:
            dataset (tf.data.TFRecordDataset): tensorflow dataset constructed from tfrecord objects
    """
    # check split name
    if split not in ['train', 'val']:
        raise ValueError('data split name {} not recognized'.format(split))
    # define dataset
    dataset = tf.data.TFRecordDataset(glob.glob(os.path.join(path_to_tfrecords, split + '-*')))
    # define constant argument to pass as image_size parameter
    image_size = tf.constant(i_size, dtype=tf.int32)
    # decode tfrecords to dataset
    dataset = dataset.map(lambda x: _parse_function(x, image_size), num_parallel_calls=n_jobs)

    return dataset


def _reshape_example(labels, N_CLASSES=21):
    """Reshapes groundtruth mask for one element of the batch as follows: each pixel is a vector with all zeros
        except for class index (e.g. [0.0, 0.0, 1.0, 0.0] for N_CLASSES=4 and class=2).

        Args:
            labels (tf.Tensor): Tensor with groundtruth mask with class numbers. Size is (width, height)
            N_CLASSES (int32): number of classes in "labels", classes are to be in range(0, N_CLASSES)
        Returns:
            labels_3d (tf.Tensor): reshaped labels Tensor
    """
    classes = range(0, N_CLASSES)
    labels_3d = list(map(lambda x: tf.equal(labels, x), classes))
    labels_3d = tf.stack(labels_3d, axis=2)
    labels_3d = tf.to_float(labels_3d)

    return labels_3d


def reshape_and_filter_batch(logits, labels, IGN_LABEL=255, N_CLASSES=21):
    """Removes IGN_LABEL from labels and corresponding entries from logits. Returns reshaped logits and labels -
       target shape is (n_valid_entries, N_CLASSES). Logits for each pixel (logits[i, j, k]) are considered valid
       if corresponding element in labels (labels[i, j, k] is not equal to IGN_LABEL, all valid entries in logits
       are stacked at 0 axis. Labels tensor is transformed as follows: for each valid entry in logits
       corresponding labels are a vector with all zeros except for class index (e.g. [0.0, 0.0, 1.0, 0.0] for
       N_CLASSES=4 and class=2)

        Args:
            logits (tf.Tensor):  Tensor with logits after forward pass of the network.
                Size is (batch_size, width, height, N_CLASSES)
            labels (tf.Tensor): Tensor with groundtruth mask with class numbers for elements in "logits".
                Size is (batch_size, width, height)
            IGN_LABEL (int32, optional): special label to ignore in "labels"
            N_CLASSES (int32, optional): number of classes in "logits" and "labels", classes are to be in range(0, N_CLASSES)
        Returns:
            new_logits (tf.Tensor):  Tensor with logits after filtering. Size is (n_valid_entries, N_CLASSES)
            new_labels (tf.Tensor): Tensor with groundtruth labels after filtering.
                Size is (n_valid_entries, N_CLASSES)
    """
    # reshape labels to binary for each class (one chanel per class)
    labels_3d = tf.map_fn(fn=lambda x: _reshape_example(x, N_CLASSES), elems=labels, dtype=tf.float32)
    # filter out all pixels in logits and labels with label == 255 (ambigious)
    filter_mask = tf.not_equal(labels, IGN_LABEL)
    # convert mask to indices
    filter_indicies = tf.to_int32(tf.where(filter_mask))
    # get elements from labels and logits by indices in filter_indices
    new_labels = tf.gather_nd(labels_3d, filter_indicies)
    new_logits = tf.gather_nd(logits, filter_indicies)

    return new_logits, new_labels
