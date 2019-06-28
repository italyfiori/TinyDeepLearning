from loss import *
from optimizer import *
from layer import *
from tensor import *
from util import *
import numpy as np
from copy import copy


class Sequential(object):
	def __init__(self, optimizer, loss, validation_data=None):
		assert isinstance(optimizer, Optimizer)
		assert isinstance(loss, Loss)
		
		self.layers = []
		self.optimizer = optimizer
		self.loss = loss
		self.validation_data = validation_data
		self.parameters = []
		
	def add(self, layer):
		"""
		Sequential网络添加layer
		"""
		assert isinstance(layer, Layer)
		self.layers.append(layer)
		
	def build_network(self):
		"""
		构建Sequential网络(初始化layers参数)
		"""
		last_output_shape = None
		self.parameters = []
		for i, layer in enumerate(self.layers):
			assert isinstance(layer, Layer)
			if i == 0:
				assert layer.get_input_shape() is not None
			else:
				layer.set_input_shape(last_output_shape)
			last_output_shape = layer.get_output_shape()
			
		self.optimizer.set_layers(self.layers)
					
	def forward(self, input_data):
		last_output_data = None
		for i, layer in enumerate(self.layers):
			if i == 0:
				last_output_data = layer.forward(input_data)
			else:
				last_output_data = layer.forward(last_output_data)		

		return last_output_data
	
	def fit(self, input_data, output_data, n_epochs, batch_size=64):
		self.build_network()
		accs = []
		errs = []
		for i in range(n_epochs):
			for x, y in get_batch_data([input_data, output_data], batch_size):
				x = Tensor(x, autograd=True)
				y = Tensor(y, autograd=True)
				y_pred = self.forward(x)
				batch_loss, acc = self.loss.back_propagate(y, y_pred)
				self.optimizer.update_layers()
				s = "\rProgress[{}], loss[{}], acc[{}%]              ".format(i, float(batch_loss), acc*100)
				accs.append(acc)
				errs.append(batch_loss)
				
				print(s, end="", flush=True)
		return errs, accs
		
		
	def predict(self, x):
		x = Tensor(x, autograd=True)
		return self.forward(x).data
		
	def summary(self):
		self.build_network()
		for i, layer in enumerate(self.layers):
			print(layer.get_summary())
