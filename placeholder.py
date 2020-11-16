import numpy as np
from tensor import *


class Input(object):
	"""
	Placeholder for graph model
	"""
	def __init__(self, shape=None, depend_placeholders=None, input_layer=None):
		self.id = None
		self.tensor = None
		self.shape = shape
		self.depend_placeholders = depend_placeholders
		self.input_layer = input_layer
		
	def set_id(self, id):
		self.id = id
		
	def set_tensor(self, tensor):
		self.tensor = tensor
		
	def set_shape(self, shape):
		self.shape = shape
		
	def set_depend_placeholders(self, depend_placeholders):
		self.depend_placeholders = depend_placeholders
		
	def set_input_layer(self, input_layer):
		self.input_layer = input_layer
	