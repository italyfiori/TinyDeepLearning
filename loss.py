#!/usr/bin/python

import numpy as np
from tensor import Tensor
from util import *

class Loss(object):
	def back_propagate(self, y, y_pred):
		assert isinstance(y, Tensor)
		assert isinstance(y_pred, Tensor)
		assert y.data.shape == y_pred.data.shape
		
	def acc(self, y, p):
		pass
		
class MSE(Loss):		
	def back_propagate(self, y, y_pred):
		"""
		从损失函数开始反向传播
		"""
		super().back_propagate(y, y_pred)
		
		loss = (y_pred - y).power(2).sum(0)
		err = loss.data.sum()
		
#		loss.refresh_dependencies()
		loss.backward(Tensor(np.ones_like(loss.data)))
		
		return err, 0
		
class SoftmaxCrossEntropy(Loss):
	def back_propagate(self, y, y_pred):
		super().back_propagate(y, y_pred)

		y_hat = softmax(y_pred.data)
		
		loss = y_pred.cross_entropy(y)
		err = float(loss.data.sum())
		acc = self.acc(y.data, y_hat)
		
		# todo
#		loss.refresh_dependencies()
		loss.backward(Tensor(np.ones_like(loss.data)))
		return err, acc
	
	def acc(self, y, p):
		y_arg = np.argmax(y, axis=1)
		p_arg = np.argmax(p, axis=1)
		
		return np.sum(y_arg == p_arg) / len(y_arg)