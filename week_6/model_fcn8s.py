import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


class FCN8s:
    """ FCN8-s model.

    All routines needed to build and run FCN8-s model.

    Args:
        path_to_weights (str): path to weights for initialization
        reduced (bool, optional): if True the FCN32-s model is build instead of FCN8-s

    Attributes:
        _data_dict (dict): dictionary with weights initialization
        _var_dict (dict): dictionary with all tf.Variable within the model
    """
    def __init__(self, path_to_weights, reduced=False):
        self._data_dict = np.load(path_to_weights, encoding='latin1').item()
        self.reduced = reduced
        self._var_dict = {}

    def build(self, rgb, keep_prob):
        # Convert RGB to BGR
        rgb = tf.cast(rgb, tf.float32)
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        self.bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        self.conv1_1 = self._conv2d(self.bgr, 3, 64, "conv1_1")
        self.conv1_2 = self._conv2d(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self._conv2d(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self._conv2d(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self._conv2d(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self._conv2d(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self._conv2d(self.conv3_2, 256, 256, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self._conv2d(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self._conv2d(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self._conv2d(self.conv4_2, 512, 512, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self._conv2d(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self._conv2d(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self._conv2d(self.conv5_2, 512, 512, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5')

        self.conv6 = self._conv2d(self.pool5, 512, 4096, 'fc6', k_size=7)
        self.drop6 = tf.nn.dropout(self.conv6, keep_prob=keep_prob, name='drop6')
        self.conv7 = self._conv2d(self.drop6, 4096, 4096, 'fc7', k_size=1)
        self.drop7 = tf.nn.dropout(self.conv7, keep_prob=keep_prob, name='drop7')

        self.score_32s = self._conv2d(self.drop7, 4096, 21, 'score_32s', k_size=1, is_act=False, stddev=0.03125)
        if self.reduced:
            self.up_32s = self._conv2d_t(self.score_32s, 21, 'up_32s', 32, 32, tf.shape(self.bgr))

            self.softmax = tf.nn.softmax(self.up_32s, axis=-1)

            self._data_dict = None
            return self.up_32s, self.softmax
        else:
            self.up_32s = self._conv2d_t(self.score_32s, 21, 'up_32s', 2, 2, tf.shape(self.pool4), is_bias=True)

            self.score_16s = self._conv2d(self.pool4, 512, 21, 'score_16s', k_size=1, is_act=False)
            self.fuse_16s = tf.add(self.up_32s, self.score_16s)
            self.up_16s = self._conv2d_t(self.fuse_16s, 21, 'up_16s', 2,  2, tf.shape(self.pool3))

            self.score_8s = self._conv2d(self.pool3, 256, 21, 'score_8s', k_size=1, is_act=False, stddev=1e-4)
            self.fuse_8s = tf.add(self.up_16s, self.score_8s)
            self.up_8s = self._conv2d_t(self.fuse_8s, 21, 'up_8s', 8, 8, tf.shape(self.bgr))

            self.softmax = tf.nn.softmax(self.up_8s, axis=-1)

            self._data_dict = None
            return self.up_8s, self.softmax

    def _conv2d_t(self, bottom, channels, name, k_size, stride, shape, is_act=False, is_bias=False):
        """Creates a transposed 2d convolution operation with given parameters and applies it to bottom.

            Args:
                bottom (tf.Tensor): input to transposed convolution operation
                channels (int32): number of input and output channels
                name (str): name of a layer to create transposed convolution in
                k_size (int32): kernel size of a transposed convolution
                stride (int32): stride of a transposed convolution
                shape (tf.Tensor): desired shape of the output
                is_act (bool, optional): whether to apply activation or not
                is_bias (bool, optional): whether to add bias term or not
            Returns:
                conv (tf.Tensor): result of transposed convolution operation applied to 'bottom'
        """
        with tf.variable_scope(name):
            # kernel shape
            shape = [shape[0], shape[1], shape[2], channels]
            # initialize kernel and bias (with tf.Variables)
            kernel, bias = self._get_conv_var(k_size, channels, channels, name)
            # create transposed convolution operation
            conv = tf.nn.conv2d_transpose(bottom, kernel, shape, [1, stride, stride, 1], padding='SAME')
            if is_bias:
                conv = tf.nn.bias_add(conv, bias)
            if is_act:
                conv = tf.nn.relu(conv)

            return conv

    def _conv2d(self, bottom, in_c, out_c, name, k_size=3, stride=1, stddev=0.001, is_act=True, is_bias=True):
        """Creates a 2d convolution operation with given parameters and applies it to bottom.

            Args:
                bottom (tf.Tensor): input to convolution operation
                in_c (int32): number of input channels
                out_c (int32): number of output channels
                name (str): name of a layer to create convolution in
                k_size (int32, optional): kernel size of a convolution
                stride (int32, optional): stride of a convolution
                stddev (float32, optional): stddev for initialization
                is_act (bool, optional): whether to apply activation or not
                is_bias (bool, optional): whether to add bias term or not
            Returns:
                conv (tf.Tensor): result of convolution operation applied to 'bottom'
        """
        with tf.variable_scope(name):
            # initialize kernel and bias (with tf.Variables)
            kernel, bias = self._get_conv_var(k_size, in_c, out_c, name, stddev)
            # create convolution operation
            conv = tf.nn.conv2d(bottom, kernel, [1, stride, stride, 1], padding='SAME')
            if is_bias:
                conv = tf.nn.bias_add(conv, bias)
            if is_act:
                conv = tf.nn.relu(conv)

            return conv

    def _max_pool(self, bottom, name):
        """Creates a maxpool operation (with fixed parameters) applied to 'bottom' with name 'name'.

            Args:
                bottom (tf.Tensor): input to maxpool operation
                name (str): name of maxpool operation
            Returns:
                tensor (tf.Tensor): result of maxpool applied to bottom
            """
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _get_conv_var(self, k_size, in_c, out_c, name, stddev=0.001):
        """Creates tf.Variable for convolutional layer 'name' with given parameters.

            Args:
                k_size (int32): kernel size of a convolution
                in_c (int32): number of input channels
                out_c (int32): number of output channels
                name (str): name of a layer to create convolution in
                stddev (float32, optional): stddev for initialization
            Returns:
                kernel (tf.Tensor): tf.Variable for a convolution kernel
                bias (tf.Tensor): tf.Variable for a convolution bias
        """
        # kernel for upsampling is initialized with bilinear filter, for other convolutions with tf.truncated_normal
        if name.startswith('up'):
            # initialize kernel with bilinear filter
            kernel_init = tf.constant(value=self._get_bilinear_kernel(in_c, factor=k_size), dtype=tf.float32)
        else:
            kernel_init = tf.truncated_normal([k_size, k_size, in_c, out_c], 0.0, stddev)
        # get tf.Variable for kernel
        kernel = self._get_var(kernel_init, name, 0, name + "_filters")
        # initialize bias
        bias_init = tf.truncated_normal([out_c], .0, stddev)
        # get tf.Variable for bias
        bias = self._get_var(bias_init, name, 1, name + "_biases")

        return kernel, bias

    def _get_var(self, init, name, idx, var_name):
        """Creates a tf.Variable with 'var_name' in 'name' layer with 'init' initialization (if no loaded values in
        self._data_dict are found).

            Args:
                init (tf.Tensor): value to initialize a tf.Variable
                name (str): name of a layer to create tf.Variable in
                idx (int): index so load/save value (0 for kernel, 1 for bias)
                var_name (str): name of a tf.Variable
            Returns:
                var (tf.Variable): new Variable with a given name and initialization
        """
        if self._data_dict is not None and name in self._data_dict:
            # reshape - to transform fc weights to conv
            value = np.reshape(self._data_dict[name][idx], init.get_shape().as_list())
        else:
            value = init
        # create variable
        var = tf.Variable(value, name=var_name)
        # save variable to dictionary
        self._var_dict[(name, idx)] = var

        assert var.get_shape() == init.get_shape()

        return var

    def _get_bilinear_kernel(self, channels, factor=2):
        """Returns kernel to initialize transposed convolution, which upsamples each of the channels by a given factor.

            Args:
                channels (int32): nubmer of channels to upsample
                factor (int32, optional): factor of upsampling
            Returns:
                weights (np.array): kernel to initialize transposed convolution
        """
        # size of the filter
        size = 2 * factor - factor % 2
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        # bilinear kernel
        kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weights = np.zeros((size, size, channels, channels), dtype=np.float32)

        # each channel is upsampled only to corresponding channel
        for i in range(0, channels):
            weights[:, :, i, i] = kernel
        return weights

    def save_npy(self, sess, npy_path="./weights/fcn8s_save.npy"):
        """Saves all weights to .npy file.

            Args:
                sess (tf.Session): an active instance of tf.Session to save weights from
                npy_path (str, optional): filename of .npy file to save weigths
            Returns:
                npy_path (str)
        """
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self._var_dict.items()):
            # get variable value from session
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("=> file saved", npy_path))
        return npy_path
