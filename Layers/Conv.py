import numpy as np
import math
import scipy.signal as sgl

class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        # if stride_shape is a value, it means the kernel shifts the same step in horizontal & vertical direction
        # generalization stride_shape in form (a, b)
        # a: represent the stride in horizontal direction, b: represent the stride in vertical direction
        if len(stride_shape) == 1:
            stride_shape = (stride_shape[0], stride_shape[0])
        self.stride_shape = stride_shape
        # define the size of kernel (c, m, n) c is channel, m*n is convolution range
        # if input kernel is (c, m), generalize it into (c, m, 1)
        if len(convolution_shape) == 2:
            convolution_shape = (convolution_shape[0], convolution_shape[1], 1)
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1], convolution_shape[2])
        self.bias = np.random.rand(num_kernels, 1)

    # two properties
    def get_gradient_weights(self):
        return self._gradient_weights

    def set_gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    gradient_weights = property(get_gradient_weights, set_gradient_weights)

    def get_gradient_bias(self):
        return self._gradient_bias

    def set_gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias

    gradient_bias = property(get_gradient_bias, set_gradient_bias)

    # forward
    def forward(self, input_tensor):
        # input_tensor shape (b, c, x, y). c is the num_channels, b is the num_batches
        # padded_input_tensor shape (b, c, x + 2 * floor(m/2) )
        # weight shape (H, c, m, n). H is the num_kernels
        # output shape (b, H, x', y')
        # x' = 1 + (x + 2 * floor(m/2) - m)/stride. Here x' = x
        # y' = 1 + (y + 2 * floor(n/2) - n)/stride. Here y' = y

        # generalization input_tensor
        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=1)

        # output_tensor (for sub-sampling)
        num_x = math.ceil(input_tensor.shape[2] / self.stride_shape[0])
        num_y = math.ceil(input_tensor.shape[3] / self.stride_shape[1])
        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, num_x, num_y))

        # do convolution, stride-shape is 1
        for b in range(input_tensor.shape[0]):
            for H in range(self.num_kernels):
                for c in range(input_tensor.shape[1]):
                    input_tensor[b, H, :, :] = sgl.convolve(input_tensor[b, c, :, :], self.weights[H, c, :, :], mode='same')
                input_tensor[b, H, :, :] += self.bias[H]
                # do sub-sampling for stride-shape isn't 1
                # num_x = math.ceil(input_tensor.shape[2] / self.stride_shape[0])
                # num_y = math.ceil(input_tensor.shape[3] / self.stride_shape[1])
                # for i in range(num_x):
                #     for j in range(num_y):
                #         input_tensor[b, H, i, j] = input_tensor[b, H, i * self.stride_shape[0], j * self.stride_shape[1]]

                # do it without loop
                output_tensor = input_tensor[b, H,
                                             0: input_tensor.shape[2]: self.stride_shape[0],
                                             0: input_tensor.shape[3]: self.stride_shape[1]]

        # if input_tensor is (b, c, x), change it back from (b, c, x, 1) to (b, c, x)
        output_tensor = np.reshape(output_tensor, (input_tensor[0], self.num_kernels, input_tensor.shape[2]))
        input_tensor = output_tensor
        return np.copy(input_tensor)

        # # zero padding (considering the size of kernel can be odd or even)
        # padding_x_before = math.floor((self.convolution_shape[1] / 2))
        # padding_x_after = math.floor(((self.convolution_shape[1] - 1) / 2))
        # padding_y_before = math.floor((self.convolution_shape[2] / 2))
        # padding_y_after = math.floor(((self.convolution_shape[2] -1) / 2))
        # input_tensor_pad = np.zeros((input_tensor.shape[0],
        #                              input_tensor.shape[1],
        #                              padding_x_before + padding_x_after + input_tensor.shape[2],
        #                              padding_y_before + padding_y_after + input_tensor.shape[3]))
        # Todo Padding要改，考虑奇数偶数不同情况
        # output_x = math.ceil((padding_x * 2 + input_tensor.shape[2] - self.convolution_shape[1]) / self.stride_shape[0])
        # output_y = math.ceil((padding_y * 2 + input_tensor.shape[3] - self.convolution_shape[2]) / self.stride_shape[1])
        # input_tensor_pad[:, :, padding_x: padding_x + input_tensor.shape[2], padding_y: padding_y + input_tensor.shape[3]] = input_tensor
        # for H in range(self.num_kernels):
        #     for i in range(output_x):
        #         for j in range(output_y):
        #             input_tensor[:, H, i, j] = np.sum(input_tensor_pad[:, :,
        #                                        i * self.stride_shape[0]: i * self.stride_shape[0] + self.convolution_shape[1],
        #                                        j * self.stride_shape[1]: j * self.stride_shape[1] + self.convolution_shape[2]]
        # Todo weights的维度不对
        #                                        * self.weights[H, :, :, :], axis=(1, 2, 3))
        #     # add bias
        #     input_tensor[:, H, :, :] = input_tensor[:, H, :, :] + self.bias

    # # property optimizer
    # def get_optimizer(self):
    #     return self._optimizer
    #
    # def set_optimizer(self, optimizer):
    #     self._optimizer = optimizer
    #
    # optimizer = property(get_optimizer, set_optimizer)
    #
    #
    # def backward(self, error_tensor):