from placeholder import *
from loss import *
from optimizer import *
from layer import *
from tensor import *
from util import *
import numpy as np
from copy import copy

class Model(object):
    """
    Grahp model class
    """
    def __init__(self, input_placeholder, output_placeholder):
        """
        初始化模型参数 input output和layers
        """
        self.layers = set()
        self.all_placeholders = {}
        
        self.input_placeholders = input_placeholder if isinstance(input_placeholder, list) else [input_placeholder]
        self.output_placeholders = input_placeholder if isinstance(output_placeholder, list) else [output_placeholder]
        
        if any(map(lambda x: not isinstance(x, Input,), self.input_placeholders)):
            raise NameError('input placeholder is not Input type!')
        if any(map(lambda x: not isinstance(x, Input,), self.output_placeholders)):
            raise NameError('output placeholder is not Input type!')
        
                
    def find_placeholders_and_layers(self):
        """
        从输出层遍历所有placeholders和layers, 保存并设置id
        """
        all_placeholders = {}
        queue = []

        for placeholder in self.output_placeholders:
            """
            从输出层回溯(添加到queue)
            """
            id = uniq_id('placeholder')
            placeholder.set_id(id)
            all_placeholders[id] = placeholder
            queue.append(placeholder)
        
        while len(queue) > 0:
            current_placeholder = queue[0]
            depend_placeholders = current_placeholder.depend_placeholders
            
            # 获取所有layers
            if current_placeholder.input_layer is not None and current_placeholder.input_layer not in self.layers:
                self.layers.add(current_placeholder.input_layer)
            
            # 回溯到起点了
            if depend_placeholders is None:
                queue.pop(0)
                continue
            
            for depend_placeholder in depend_placeholders:
                # 已经添加过得placeholder不再处理
                if depend_placeholder.id is not None:
                    continue
                
                # 获取所有placeholder
                placeholder_id = uniq_id('placeholder')
                depend_placeholder.set_id(placeholder_id)
                all_placeholders[placeholder_id] = depend_placeholder
                queue.append(depend_placeholder)
            
            # 所有输入placeholder都获取到了, 处理下一个
            queue.pop(0)
        
        # 获取所有placeholders
        self.all_placeholders = all_placeholders
        
    def forward(self, input_datas):
        """
        从输入层遍历计算所有placeholder的tensor
        """
        ready_placeholders_ids = set()
        waiting_placeholders_ids = set(self.all_placeholders.keys())
        
        # 从输入层开始, 给输入节点赋值
        for i, placeholder in enumerate(self.input_placeholders):
            placeholder.set_tensor(input_datas[i])
            ready_placeholders_ids.add(placeholder.id)
            waiting_placeholders_ids.remove(placeholder.id)
            
        # 遍历计算所有placeholder
        while len(waiting_placeholders_ids) > 0:
            next_placeholder = None
            for waiting_placeholder_id in waiting_placeholders_ids:
                waiting_placeholder = self.all_placeholders[waiting_placeholder_id]
                depend_placeholders = waiting_placeholder.depend_placeholders
                
                # 需要所有依赖的上游节点都ready
                if any(map(lambda x: x.id not in ready_placeholders_ids, depend_placeholders)):
                    continue
                
                # 当前节点的依赖
                depend_tensors = [depend_placeholder.tensor for depend_placeholder in depend_placeholders]
                
                # 计算当前节点的值
                if len(depend_tensors) == 1:
                    output_tensor = waiting_placeholder.input_layer.forward(depend_tensors[0])
                elif len(depend_tensors) == 2:
                    output_tensor = waiting_placeholder.input_layer.forward(depend_tensors[0], depend_tensors[1], )
                else:
                    raise("depend tensors numbuer should be 1 or 2!")
                
                
                waiting_placeholder.set_tensor(output_tensor)
                next_placeholder = waiting_placeholder
                break
                
            # 已经计算出结果的节点放入ready集合
            if next_placeholder is not None:
                ready_placeholders_ids.add(waiting_placeholder.id)
                waiting_placeholders_ids.remove(waiting_placeholder.id)
            
        
    def compile(self, optimizer, loss, validation_data=None):
        """
        设置模型优化方法、损失函数和校验数据集
        """
        assert isinstance(optimizer, Optimizer)
        assert isinstance(loss, Loss)
        
        self.optimizer = optimizer
        self.loss = loss
        self.validation_data = validation_data
        
        self.find_placeholders_and_layers()
        self.optimizer.set_layers(self.layers)
        
    def get_batch_data(self, data):
        return data
        
    def fit(self, input_data, output_data, n_epochs, batch_size=None):
        """
        训练模型
        """
        # 检查输入数据并转换成Tensor格式
        input_datas = input_data if isinstance(input_data, list) else [input_data]
        output_datas = output_data if isinstance(input_data, list) else [output_data]
        
        if any(map(lambda x: not isinstance(x, np.ndarray), input_datas)):
            raise NameError('input placeholder is not Numpy.Array type!')
        if any(map(lambda x: not isinstance(x, np.ndarray,), output_datas)):
            raise NameError('output placeholder is not Numpy.Array type!')
            
        # 初始化模型所有layer的参数
        metrices = []
        accs = []
        errs = []
        for i in range(n_epochs):
            for batch_data in get_batch_data(input_datas + output_datas):
                batch_input_tensors = list(map(lambda x: Tensor(x, autograd=True), batch_data[:len(input_datas)]))
                batch_output_tensors = list(map(lambda x: Tensor(x, autograd=True), batch_data[len(input_datas):]))
                
                # 前向遍历
                self.forward(batch_input_tensors)
                
                # todo 暂时只支持单tensor输出
                assert len(self.output_placeholders) == 1
                assert len(batch_input_tensors) == 1
                
                # 从损失函数结果反向传递梯度
                err, acc = self.loss.back_propagate(batch_output_tensors[0], self.output_placeholders[0].tensor)
                errs.append(err)
                accs.append(acc)
                # 更新模型所有参数
                self.optimizer.update_layers()
                
                s = "\rProgress[{}], error[{}], acc[{}%]              ".format(i, float(err), acc*100)
                print(s, end="", flush=True)
        return errs, accs
    