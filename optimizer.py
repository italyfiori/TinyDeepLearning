from tensor import Tensor
import numpy as np

class Optimizer(object):
	def set_layers(self, layers):
		self.layers = layers
		
	def update_layers(self):
		pass

class SGD(Optimizer):
	"""
	使用sgd更新需要更新的参数
	"""
	
	def __init__(self, learning_rate = 0.01):
		self.learning_rate = learning_rate
		
	def update_layers(self):
		for layer in self.layers:
			for param in layer.parameters:
				try:
					param.data -= param.grad.data * self.learning_rate
					param.grad.data *= 0
				except Exception as e:
					print(e)
					print('op params', param.op, param.data.shape)
					print('children', param.children)
					exit()
					


class Momentum(Optimizer):
	def __init__(self, learning_rate = 0.01, momentum=0.9):
		self.v = None
		self.learning_rate = learning_rate
		self.momentum = momentum
	
	def update_layers(self):
		# 保存momentum历史记录
		if self.v is None:
			self.v = {}
			for i, layer in enumerate(self.layers):
				self.v[i] = {}
				for j, param in enumerate(layer.parameters):
					self.v[i][j] = np.zeros_like(param.data)
		
		# 更新参数
		for i, layer in enumerate(self.layers):
			for j, param in enumerate(layer.parameters):
				self.v[i][j] = self.momentum * self.v[i][j] + (1 - self.momentum) * param.grad.data
				param.data -= self.learning_rate * self.v[i][j]
				param.grad.data *= 0
				
	
		
class RMSprop(Optimizer):
	def __init__(self, learning_rate=0.01, decay_rate=0.99):
		self.h = None
		self.learning_rate = learning_rate
		self.decay_rate = decay_rate
		
	def update_layers(self):
		# 保存momentum历史记录
		if self.h is None:
			self.h = {}
			for i, layer in enumerate(self.layers):
				self.h[i] = {}
				for j, param in enumerate(layer.parameters):
					self.h[i][j] = np.zeros_like(param.data)
					
		
		for i, layer in enumerate(self.layers):
			for j, param in enumerate(layer.parameters):
				self.h[i][j] = self.decay_rate * self.h[i][j]  + (1 - self.decay_rate) * np.power(param.grad.data, 2)
				param.data -= self.learning_rate * param.grad.data / (np.sqrt(self.h[i][j])	+ 1e-7)
				
				param.grad.data *= 0


class Adam(Optimizer):
	def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999):
		self.m = None
		self.v = None
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.iter = 0
		
	def update_layers(self):
		# 保存momentum历史记录
		if self.v is None:
			self.m = {}
			self.v = {}
			for i, layer in enumerate(self.layers):
				self.m[i] = {}
				self.v[i] = {}
				for j, param in enumerate(layer.parameters):
					self.m[i][j] = np.zeros_like(param.data)
					self.v[i][j] = np.zeros_like(param.data)
		
		self.iter += 1
		lr_t  = self.learning_rate * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)    
			
		for i, layer in enumerate(self.layers):
			for j, param in enumerate(layer.parameters):
				self.m[i][j] = self.beta1 * self.m[i][j] + (1 - self.beta1) * (param.grad.data)
				self.v[i][j] = self.beta2 * self.v[i][j] + (1 - self.beta2) * (param.grad.data**2)
				param.data -= self.learning_rate * self.m[i][j] / (np.sqrt(self.v[i][j]) + 1e-7)
				
				param.grad.data *= 0