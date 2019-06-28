import numpy as np
from tensor import *
from util import *
from placeholder import *


class Layer(object):

    def __init__(self):
        self.parameters = []
        self.input_datas = None
        self.input_shapes = None
        self.output_shape = None

    def add_parameter(self, param):
        """
        添加lyaer需要update的参数, 如weights bias
        """
        param.set_trainable(True)
        self.parameters.append(param)

    def get_parameters(self):
        """
        获取layer所有需要update的参数
        """
        return self.parameters

    def set_input_shape(self, input_shape):
        """
        设置layer输入数据的shape(不报包含batch_size)
        """
        self.input_shapes = [input_shape]

    def get_input_shape(self):
        """
        获取layer的输入shape
        """
        return self.input_shapes[0]

    def set_output_shape(self, output_shape):
        """
        获取layer输出数据的shape(不包含batch_size)
        """
        self.output_shape = output_shape

    def get_output_shape(self):
        """
        获取layer的输出shape
        """
        pass

    def init_params(self):
        """
        初始化layer的参数, 交给子类实现
        """
        pass

    def __call__(self, input_placeholder):
        """
        ()运算符, 用于model的函数式api
        """
        assert isinstance(input_placeholder, Input)

        # input_placeholder是当前layer的输入
        self.set_input_shape(input_placeholder.shape)

        # output_placeholder的shape当前layer的输出shape
        output_shape = self.get_output_shape()

        # 创建输出placeholder
        return Input(output_shape, [input_placeholder], self)


class Activation(Layer):
    def __init__(self, activation_type):
        super().__init__()
        self.activation_type = activation_type

    def get_output_shape(self):
        self.set_output_shape(self.get_input_shape())
        return self.output_shape

    def forward(self, input_tensor):
        """
        前向计算, 用于model和sequential
        """
        if self.activation_type == 'relu':
            return input_tensor.relu()
        elif self.activation_type == 'sigmoid':
            return input_tensor.sigmoid()
        elif self.activation_type == 'softmax':
            return input_tensor.softmax()

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        
    def get_output_shape(self):
        input_shape = self.get_input_shape()
        output_shape = (input_shape[0], np.prod(input_shape[1:]))
        self.set_output_shape(output_shape)
        return output_shape

    def forward(self, input_tensor):
        """
        前向计算, 用于model和sequential
        """
        return input_tensor.flatten()

class Conv2D(Layer):
    def __init__(self, n_filters, filter_shape, stride=1, input_shape=None, pad=1):
        super().__init__()
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.stride = stride
        self.pad = pad

        if input_shape is not None:
            C, H, W = input_shape
            self.set_input_shape((None, C, H, W))

        self.filter_weight = None  # (C*FH*FW, FN)
        self.filter_bias = None  # (FN,)

    def get_output_shape(self):
        N, C, H, W = self.get_input_shape()
        OH, OW = get_conv_output_shape(H, W, self.filter_shape, self.stride, self.pad)
        output_shape = (N, self.n_filters, OH, OW)
        self.set_output_shape(output_shape)
        return output_shape
    
    def init_params(self):
        FH, FW = self.filter_shape
        C = self.get_input_shape()[1]
        
        weights_data = np.random.randn(C * FH * FW, self.n_filters) * np.sqrt(2.0 / C * FH * FW)
        bias_data = np.zeros(self.n_filters)
        self.filter_weight = Tensor(weights_data, autograd=True)  # (C*FH*FW, FN)
        self.filter_bias = Tensor(bias_data, autograd=True)  # (FN,)
        
        self.add_parameter(self.filter_weight)
        self.add_parameter(self.filter_bias)

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_data : 输入数据的形状 (N C H W)
        Returns (N C OH OW)
        -------
        """
        N, C, H, W = input_tensor.data.shape
        OH, OW = get_conv_output_shape(H, W, self.filter_shape, self.stride, self.pad)
        if self.filter_weight is None:
            self.set_input_shape((N, C, H, W))
            self.get_output_shape()
            self.init_params()

        col = input_tensor.im2col(self.filter_shape, self.stride, self.pad)  # (N*OH*OW, C*FH*FW)
        out = col.dot(self.filter_weight)  # (N*OH*OW, FN)
        out = out + self.filter_bias.expand(0, col.data.shape[0])  # (N*OH*OW, FN)
        return out.reshape((N, OH, OW, -1)).transpose((0, 3, 1, 2)) # （N， n_filter, OH, HW）

class MaxPooling(Layer):
    def __init__(self, pool_shape, stride=1, input_shape=None, pad=0):
        super().__init__()
        
        self.pool_shape = pool_shape
        self.stride = stride
        self.input_shape = input_shape
        self.pad = pad

        if input_shape is not None:
            C, H, W = input_shape
            self.set_input_shape((None, C, H, W))

    def get_output_shape(self):
        N, C, H, W = self.get_input_shape()
        OH, OW = get_conv_output_shape(H, W, self.pool_shape, self.stride, self.pad)
        output_shape = (N, C, OH, OW)
        self.set_output_shape(output_shape)
        return output_shape

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_data : 输入数据的形状 (N C H W)
        Returns (N C OH OW)
        -------
        """
        self.set_input_shape(input_tensor.data.shape)
        self.get_output_shape()

        N, C, H, W = input_tensor.data.shape
        OH, OW = get_conv_output_shape(H, W, self.pool_shape, self.stride, self.pad)

        col = input_tensor.im2col(self.pool_shape, self.stride, self.pad)
        col = col.reshape((-1, self.pool_shape[0] * self.pool_shape[1]))
        out = col.max()
        return out.reshape((N, OH, OW, C)).transpose((0, 3, 1, 2))


class Dense(Layer):
    def __init__(self, output_dim, input_shape=None):
        super().__init__()
        self.output_dim = output_dim
        self.weights = None
        self.bias = None
        
        if input_shape is not None:
            self.set_input_shape((None,) + input_shape)
        
    def get_output_shape(self):
        input_shape = self.get_input_shape()
        output_shape = input_shape[:-1] + (self.output_dim, )
        self.set_output_shape(output_shape)
        return self.output_shape

    def init_params(self):
        input_dim = self.get_input_shape()[-1]
        output_dim = self.get_output_shape()[-1]

        weights_data = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        bias_data = np.zeros(output_dim)

        self.weights = Tensor(weights_data, autograd=True)
        self.bias = Tensor(bias_data, autograd=True)

        self.parameters = []
        self.add_parameter(self.weights)
        self.add_parameter(self.bias)

    def forward(self, input_tensor):
        """
        前向计算, 用于model和sequential
        """
        if self.weights is None:
            self.set_input_shape(input_tensor.data.shape)
            self.get_output_shape()
            self.init_params()
        
        if len(input_tensor.data.shape) == 2:
            batch_size, _ = input_tensor.data.shape
            return input_tensor.dot(self.weights) + self.bias.expand(0, batch_size)
        elif len(input_tensor.data.shape) == 3:
            batch_size, times, _ = input_tensor.data.shape
            i = input_tensor.dot(self.weights)
            t = self.bias.expand(0, times).expand(0, batch_size)
            return i + t

            
class RNN(Layer):
    def __init__(self, n_units, input_shape=None, activation='tanh', return_type='sequence'):
        super().__init__()
        assert activation in ['relu', 'sigmoid', 'tanh']
        assert return_type in ['sequence', 'states', 'last_state', 'last_output']
        
        self.n_units = n_units
        self.activation = activation
        self.return_type = return_type
        
        if input_shape is not None:
            self.set_input_shape( (None,) + input_shape )
        
        self.U = None # input weight
        self.V = None # output weight
        self.W = None # state weight
        
    def get_output_shape(self):
        batch_size, times, input_dim = self.get_input_shape()
        
        # 1初始化参数 并计算output_shape
        if self.return_type == 'sequence':
            output_shape = (batch_size, times, input_dim)
        elif self.return_type == 'states':
            output_shape = (batch_size, times, self.n_units)
        elif self.return_type == 'last_state':
            output_shape = (batch_size, self.n_units)
        elif self.return_type == 'last_output':
            output_shape = (batch_size, input_dim)

        self.set_output_shape(output_shape)
        return output_shape
        
    def init_params(self):
        _, _, input_dim = self.get_input_shape()
        
        if self.return_type in ['sequence', 'last_output']:
            V_data = np.random.randn(self.n_units, input_dim) * np.sqrt(2.0 / self.n_units)
            self.V = Tensor(V_data, autograd=True)
            self.add_parameter(self.V)
        
        U_data = np.random.randn(input_dim, self.n_units) * np.sqrt(2.0 / input_dim)
        W_data = np.random.randn(self.n_units, self.n_units) * np.sqrt(2.0 / self.n_units)

        self.U = Tensor(U_data, autograd=True)
        self.W = Tensor(W_data, autograd=True)

        self.add_parameter(self.U)
        self.add_parameter(self.W)
        
    def forward(self, input_tensor, init_state=None):
        batch_size, times, input_dim = input_tensor.data.shape
        
        # 1初始化参数 并计算output_shape
        if self.U is None:
            self.set_input_shape(input_tensor.data.shape)
            self.get_output_shape()
            self.init_params()
        
        # 2 计算states
        states = []
        prev_state = init_state
        for i in range(times):
            cur_input = input_tensor.select_index(1, i) # i时刻的输入 (batch_size, input_dim)
            cur_state = self.step(cur_input, prev_state) # i时刻的状态 (batch_size, n_units)
            prev_state = cur_state
            states.append(cur_state)
        
        # 3 计算返回结果
        if self.return_type == 'sequence':
            outputs = []
            for i in range(times):
                outputs.append(states[i].dot(self.V))
            outputs = [output.reshape((batch_size, 1, input_dim)) for output in outputs]
            return Tensor.concatenate(outputs, 1)
        elif self.return_type == 'states':
            states = [state.reshape((batch_size, 1, self.n_units)) for state in states]
            return Tensor.concatenate(states, 1)
        elif self.return_type == 'last_output':
            return states[-1].dot(self.V)
        elif self.return_type == 'last_state':
            return states[-1]

        raise("rnn code shold not come here!")
            
    def step(self, cur_input, prev_state=None):
        if prev_state is None:
            batch_size, input_dim = cur_input.data.shape
            prev_state = Tensor(np.zeros((batch_size, self.n_units)), autograd=True)
        return (cur_input.dot(self.U) + prev_state.dot(self.W)).activation(self.activation)
        
    def __call__(self, input_placeholder, state_placeholder=None):
        """
        ()运算符, 用于函数式api
        """
        assert isinstance(input_placeholder, Input)
        assert len(input_placeholder.shape) == 3

        # input_placeholder是当前layer的输入
        self.set_input_shape(input_placeholder.shape)
        output_shape = self.get_output_shape()

        # 创建output_placeholder
        if state_placeholder is not None:
            assert isinstance(state_placeholder, Input)
            output_placeholder = Input(output_shape, [input_placeholder, state_placeholder], self)
        else:
            output_placeholder = Input(output_shape, [input_placeholder], self)
        return output_placeholder
        

class Dropout(Layer):
    def __init__(self, ratio=0.5):
        super().__init__()
        self.ratio = ratio

    def get_output_shape(self):
        output_shape = self.get_input_shape()
        self.set_output_shape(output_shape)
        return output_shape
        
    def forward(self, input_tensor):
        return input_tensor.dropout(self.ratio)

class Reshape(Layer):
    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape
        
    def get_output_shape(self):
        self.set_output_shape(self.new_shape)
        return self.output_shape

    def forward(self, input_tensor):
        return input_tensor.reshape(self.new_shape)
        

class Add(Layer):
    def __init__(self):
        super().__init__()
        
    def get_output_shape(self):
        output_shape = self.get_input_shape()
        self.set_output_shape(output_shape)
        return output_shape

    def forward(self, input_tensor, other_tensor):
        return input_tensor.__add__(other_tensor)
        
    def __call__(self, input_placeholder, other_placeholder):
        assert isinstance(input_placeholder, Input)
        assert input_placeholder.shape == other_placeholder.shape
        
        # input_placeholder是当前layer的输入
        self.set_input_shape(input_placeholder.shape)
        output_shape = self.get_output_shape()

        # 创建输出placeholder
        return Input(output_shape, [input_placeholder, other_placeholder], self)
