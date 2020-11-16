from util import *
import numpy as np

class Tensor(object):
    """
    Basic tensor compute function 
    """
    def set_trainable(self, trainable):
        self.trainable = trainable
        
    def __init__(self, data, parents=None, op=None, autograd=False, id=None):
        """
        初始化Tensor
        """
        self.data = np.array(data)
        self.parents = parents # 生成当前tensor的源tensor
        self.op = op # 生成当前tensor的计算方法
        self.op_params = None
        self.autograd = autograd
        self.trainable = False
        if id is None:
            id = uniq_id('tensor')
        self.id = id
        
        # 梯度初始化
        self.grad = None
        self.momentum = None
        self.children = {}
        
        # 构造Tensor关联关系(parents和children)
        if parents is None:
            return

        for parent in parents:
            if self.id not in parent.children:
                parent.children[self.id] = 0
            parent.children[self.id] += 1
            
    def has_receive_all_children_gradients(self):
        """
        当前节点是否接收了所有下游节点反向传播的梯度
        """
        for child_id, child_cnt in self.children.items():
            if child_cnt > 0:
                return False
        return True
        

    def backward(self, child_grad=None, child_node=None):
        """
        梯度反向传播
        """
        # 当前节点不能反向传播
        if not self.autograd:
            del self
            return 
        
        # 1. 下游节点的梯度汇总传播到当前节点
        assert isinstance(child_grad, Tensor)
        if self.grad is None:
            self.grad = Tensor(child_grad.data)
        else: 
            self.grad.data += child_grad.data
        
        # 下游节点反向传播次数计数减1
        if child_node is not None:
            self.children[child_node.id] -= 1
            assert self.children[child_node.id] >= 0
        
        # 没有上游节点, 则不需要再向上游反向传播
        if self.parents is None:
            return 
        
        # 有下游节点且下游节点的梯度没有全部传播到当前节点，则当前节点还不能向后传播梯度
        if child_node is not None and not self.has_receive_all_children_gradients():
            return

        # 2. 当前节点继续上游反向传播
        if self.op == 'add':
            grad = Tensor(self.grad.data)
            self.parents[0].backward(grad, self)
            self.parents[1].backward(grad, self)
        
        elif self.op == 'sub':
            grad1 = Tensor(self.grad.data)
            grad2 = Tensor(-self.grad.data)
            self.parents[0].backward(grad1, self)
            self.parents[1].backward(grad2, self)
            
        elif self.op == 'neg':
            grad = Tensor(-self.grad.data)
            self.parents[0].backward(grad, self)

        elif self.op == 'mul':
            grad1 = Tensor(self.grad.data * self.parents[1].data)
            grad2 = Tensor(self.grad.data * self.parents[0].data)
            self.parents[0].backward(grad1, self)
            self.parents[1].backward(grad2, self)
            
        elif self.op == 'dot':
            grad1 = Tensor(self.grad.data.dot(self.parents[1].data.transpose()))
            
            if len(self.parents[0].data.shape) == 2:
                grad2 = Tensor(self.parents[0].data.transpose().dot(self.grad.data))
            elif len(self.parents[0].data.shape) == 3:
                # 支持3维数据输入
                batch_size, times, input_dim = self.parents[0].data.shape
                _, _, output_dim = self.grad.data.shape
                _input_data = self.parents[0].data.transpose(-1, 0, 1).reshape((input_dim, -1))
                _grad_data = self.grad.data.reshape((-1, output_dim))
                grad2 = Tensor(_input_data.dot(_grad_data))
            else:
                raise("input data dimension should be 2 or 3!")
            
            self.parents[0].backward(grad1, self)
            self.parents[1].backward(grad2, self)
            
        elif self.op == 'transpose':
            if self.op_params['axes'] is not None:
                axes = self.op_params['axes']
                revert_axes = tuple([axes.index(i) for i in range(len(axes))])
                grad = Tensor(self.grad.data.transpose(revert_axes))
            else:
                grad = Tensor(self.grad.data.transpose())
            self.parents[0].backward(grad, self)
            
        elif self.op == 'reshape':
            old_shape = self.parents[0].data.shape
            grad = Tensor(self.grad.data.reshape(old_shape))
            self.parents[0].backward(grad, self)
            
        elif self.op == 'sum':
            dim = self.op_params['dim']
            copies = self.parents[0].data.shape[dim]
            grad_data = expand_data(self.grad.data, dim, copies)
            self.parents[0].backward(Tensor(grad_data), self)
            
        elif self.op == 'max':
            arg_max = self.op_params['arg_max']
            data_shape = self.parents[0].data.shape
            grad_data = np.zeros(data_shape)
            grad_data[np.arange(data_shape[0]), arg_max] = self.grad.data
            self.parents[0].backward(Tensor(grad_data), self)
        
        elif self.op == 'expand':
            dim = self.op_params['dim']
            grad = Tensor(self.grad.data.sum(dim))
            self.parents[0].backward(grad, self)
            
        elif self.op == 'relu':
            grad_data = self.grad.data * ((self.parents[0].data > 0 ) * np.ones_like(self.parents[0].data))
            self.parents[0].backward(Tensor(grad_data), self)
            
        elif self.op == 'sigmoid':
            grad_data = self.grad.data * self.data * (1 - self.data)
            self.parents[0].backward(Tensor(grad_data), self)
            
        elif self.op == 'tanh':
            grad_data =  self.grad.data * (1 - np.power(self.data, 2))
            self.parents[0].backward(Tensor(grad_data), self)
        
        elif self.op == 'softmax':
            grad = Tensor(self.data * (1 - self.data))
            self.parents[0].backward(grad, self)
        
        elif self.op == 'flatten':
            grad = Tensor(self.grad.data.reshape(self.parents[0].data.shape))
            self.parents[0].backward(grad, self)
            
        elif self.op == 'im2col':
            dcol = self.grad.data 
            input_shape = self.parents[0].data.shape
            filter_shape = self.op_params['filter_shape']
            stride = self.op_params['stride']
            pad = self.op_params['pad']

            grad_data = col2im(dcol, input_shape, filter_shape, stride, pad)
            self.parents[0].backward(Tensor(grad_data), self)
            
        elif self.op == 'dropout':
            grad_data = self.grad.data * self.op_params['mask']
            self.parents[0].backward(Tensor(grad_data), self)
            
        elif self.op == 'cross_entropy':
            p = self.op_params['p']
            y = self.parents[1].data # target
            p = np.clip(p, 1e-15, 1 - 1e-15)
            grad_data = (p - y) / p.shape[0]
            
            self.parents[0].backward(Tensor(grad_data), self)
            
        elif self.op == 'select_index':
            grad_data = np.zeros_like(self.parents[0].data).astype('float64')
            array_index_plus(grad_data, self.op_params['dim'], self.op_params['i'], self.grad.data)
            
            self.parents[0].backward(Tensor(grad_data), self)
            
        elif self.op == 'power':
            num = self.op_params['num']
            grad_data = self.grad.data * num * np.power(self.parents[0].data, num - 1)
            
            self.parents[0].backward(Tensor(grad_data), self)
            
        elif self.op == 'concatenate':
            axis = self.op_params['axis']
            lens = [parent.data.shape[axis] for parent in self.parents]
            idxs = np.cumsum(lens)
            
            for i in range(len(self.parents)):
                select_range = range(0, idxs[i]) if i == 0 else range(idxs[i-1], idxs[i])
                grad_data = array_index_select(self.grad.data, axis, select_range)
                self.parents[i].backward(Tensor(grad_data), self)
        
        if not self.trainable:
            del self
        
    def __add__(self, other):
        """
        加法运算
        """
        return Tensor(self.data + other.data, parents=[self, other], op='add', autograd=True)
        
    def __sub__(self, other):
        """
        减法运算
        """
        return Tensor(self.data - other.data, parents=[self, other], op='sub', autograd=True)
        
    def __neg__(self):
        """
        取反运算
        """
        return Tensor(self.data * -1, parents=[self], op='neg', autograd=True)
        
    def __mul__(self, other):
        """
        矩阵元素乘
        """
        return Tensor(self.data * other.data, parents=[self, other], op='mul', autograd=True)
        
    def dot(self, other):
        """
        矩阵乘
        """
        return Tensor(self.data.dot(other.data), parents=[self, other], op='dot', autograd=True)
        
    def sum(self, dim):
        """
        维度求和(矩阵维度减小)
        """
        out = Tensor(self.data.sum(dim), parents=[self], op='sum', autograd=True)
        out.op_params = {'dim': dim}
        return out
        
    def max(self):
        assert len(self.data.shape) == 2
        
        max_val = self.data.max(axis=1)
        out = Tensor(max_val, parents=[self], op='max', autograd=True)
        arg_max = self.data.argmax(axis=1)
        out.op_params = {'arg_max': arg_max}
        return out
        
        
    def expand(self, dim, copies):
        """
        维度扩展(矩阵维度增大)
        """
        new_data = expand_data(self.data, dim, copies)
        out = Tensor(new_data, parents=[self], op='expand', autograd=True)
        out.op_params = {'dim': dim}
        return out
        
    def reshape(self, new_shape):
        new_data = self.data.reshape(new_shape)
        return Tensor(new_data, parents=[self], op='reshape', autograd=True)
        
    def transpose(self, axes=None):
        """
        转置运算
        """
        new_data = np.transpose(self.data, axes)
        out = Tensor(new_data, parents=[self], op='transpose', autograd=True)
        out.op_params = {'axes': axes}
        return out
        
    def relu(self):
        data = (self.data > 0) * self.data
        return Tensor(data, parents=[self], op='relu', autograd=True)
        
    def sigmoid(self):
        data = 1 / (1 + np.exp(-self.data))
        return Tensor(data, parents=[self], op='sigmoid', autograd=True)
        
    def tahn(self):
        data = 2 / (1 + np.exp(-2*self.data)) - 1
        return Tensor(data, parents=[self], op='tanh', autograd=True)
        
    def activation(self, type):
        if type == 'relu':
            return self.relu()
        elif type == 'sigmoid':
            return self.sigmoid()
        elif type == 'tanh':
            return self.tahn()
        
    def softmax(self):
        data_max = np.max(self.data, axis=-1, keepdims=True)
        data_exp = np.exp(self.data - data_max) # 防止指数计算溢出
        data_sum = np.sum(data_exp, axis=-1, keepdims=True)
        return Tensor(data_exp / data_sum, parents=[self], op='softmax', autograd=True)
        
    def cross_entropy(self, target):
        """
        计算softmax交叉熵
        """
        data_max = np.max(self.data, axis=-1, keepdims=True)
        data_exp = np.exp(self.data - data_max) # 防止指数计算溢出
        data_sum = np.sum(data_exp, axis=-1, keepdims=True)
        p = data_exp / data_sum
        y = target.data
        
        delta = 1e-7
        batch_size = self.data.shape[0]
        output_data = -np.sum(y * np.log(p + delta))
        
        out = Tensor(output_data, parents=[self, target], op='cross_entropy', autograd=True)
        out.op_params = {'p': p}
        return out
        
    def flatten(self):
        """
        数据展平
        """
        data = self.data.reshape(self.data.shape[0], -1)
        return Tensor(data, parents=[self], op='flatten', autograd=True)
        
    def im2col(self, filter_shape, stride=1, pad=0):
        """
        批量image数据按照卷积形式展开
        """
        data = im2col(self.data, filter_shape, stride, pad)
        op_params = {
            'filter_shape': filter_shape,
            'stride': stride,
            'pad': pad,
        }
        output = Tensor(data, parents=[self], op='im2col', autograd=True)
        output.op_params = op_params
        return output
        
    def dropout(self, ratio, train=True):
        """
        dropout
        """
        mask = np.random.random(self.data.shape) > ratio
        if train:
            output_data = mask * self.data
        else:
            output_data = self.data * (1.0 - ratio)
        
        if self.autograd:
            output = Tensor(output_data, parents=[self], op='dropout', autograd=True)
            output.op_params = {'mask': mask}
            return output
            
        return Tensor(output_data)
        
    def select_index(self, dim, i):
        """
        select某个维度的数据
        """
        output_data = array_index_select(self.data, dim, i)
        output = Tensor(output_data, parents=[self], op='select_index', autograd=True)
        output.op_params = {'dim': dim, 'i': i}
        return output
        
    def power(self, num):
        data = np.power(self.data, num)
        out = Tensor(data, parents=[self], op='power', autograd=True)
        out.op_params = {'num': num}
        return out
        
    @staticmethod
    def concatenate(tensors, axis):
        """
        合并数据
        """
        tensors_data = [tensor.data for tensor in tensors]
        output_data = np.concatenate(tensors_data, axis)
    
        output = Tensor(output_data, parents=tensors, op='concatenate', autograd=True)
        output.op_params = {'axis': axis}
        return output
    
    def clean_dependencies(self):
        """
        清除所有依赖计数
        """
        self.children = {}
        if self.parents is not None:
            for parent in self.parents:
                parent.clean_dependencies()
    
    def create_dependencies(self):
        """
        todo 目前计数不准, 待排查
        创建依赖计数, 只考虑下游通往loss的节点
        """
        if self.parents is not None:
            for parent in self.parents:
                if self.id not in parent.children:
                    parent.children[self.id] = 0	
                parent.children[self.id] += 1
                parent.create_dependencies()
    
    def refresh_dependencies(self):
        """
        只保留有
        """
        self.clean_dependencies()
        self.create_dependencies()
    
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())
        
        
        