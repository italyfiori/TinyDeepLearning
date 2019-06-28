import numpy as np
import matplotlib.pyplot as plt


class Util:
	"""
	获取唯一随机数
	"""
	random_sets = {}
	
	@staticmethod
	def uniq_id(group='default'):
		rand_id = np.random.randint(0, 10000000)
		if group not in Util.random_sets:
			Util.random_sets[group] = set([rand_id])
			return rand_id
			
		while rand_id in Util.random_sets[group]:
			rand_id = np.random.randint(0, 10000000)
		Util.random_sets[group].add(rand_id)
		return rand_id
	
	@staticmethod
	def clear():
		Util.random_sets = {}

def im2col(input_data, filter_shape, stride=1, pad=0):
	"""
	Parameters
	----------
	input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据 (N C H W)
	filter_shape : 卷积核的形状
	filter_h : 滤波器的高
	filter_w : 滤波器的长
	stride : 步幅
	pad : 填充
	Returns (N*OH*OW, C*FH*FW)
	-------
	col : 2维数组
	"""
	N, C, H, W = input_data.shape
	filter_h, filter_w = filter_shape
	out_h = (H + 2*pad - filter_h)//stride + 1
	out_w = (W + 2*pad - filter_w)//stride + 1

	img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
	col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

	for y in range(filter_h):
		y_max = y + stride*out_h
		for x in range(filter_w):
			x_max = x + stride*out_w
			col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

	col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
	return col


def col2im(col, input_shape, filter_shape, stride=1, pad=0):
	"""
	Parameters
	----------
	col :
	input_shape : 输入数据的形状（例：(10, 1, 28, 28)）(N*OH*OW, C*FH*FW)
	filter_shape : 卷积核的形状
	stride
	pad
	Returns (N C H W)
	-------
	"""
	N, C, H, W = input_shape
	filter_h, filter_w = filter_shape
	out_h = (H + 2*pad - filter_h)//stride + 1
	out_w = (W + 2*pad - filter_w)//stride + 1
#	print('col shape:', col.shape)
	col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

	img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
	for y in range(filter_h):
		y_max = y + stride*out_h
		for x in range(filter_w):
			x_max = x + stride*out_w
			img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
	
	return img[:, :, pad:H + pad, pad:W + pad]
	
def get_conv_output_shape(H, W, filter_shape, stride, pad):
	FH, FW = filter_shape
	OH = 1 + int((H + 2*pad - FH) / stride)
	OW = 1 + int((W + 2*pad - FW) / stride)
	return (OH, OW)
	
def to_categorical(data, col_num=None):
	col_num = col_num if col_num is not None else np.max(data) + 1
	data_size = data.shape[0]
	one_hot = np.zeros((data_size, col_num))
	one_hot[np.arange(data_size), data] = 1
	return one_hot
			
def get_batch_data(data, batch_size=64):
	n_samples = data[0].shape[0] if isinstance(data, list) else data.shape[0]
	
	for i in np.arange(0, n_samples, batch_size):
		begin, end = i, min(i+batch_size, n_samples)	
		if isinstance(data, list):
			yield tuple([x[begin:end] for x in data])
		else:
			yield data[begin:end]

def shuffle_data(X, y, seed=None):
	""" Random shuffle of the samples in X and y """
	if seed:
		np.random.seed(seed)
	idx = np.arange(X.shape[0])
	np.random.shuffle(idx)
	return X[idx], y[idx]
	
def train_test_split(X, y, test_ratio=0.5, shuffle=True, seed=None):
	""" Split the data into train and test sets """
	if shuffle:
		X, y = shuffle_data(X, y, seed)

	split_i = int(len(y) * (1 - test_ratio))
	X_train, X_test = X[:split_i], X[split_i:]
	y_train, y_test = y[:split_i], y[split_i:]
	return X_train, X_test, y_train, y_test
	

def plot(data):
	x = np.arange(0, len(data))
	y = np.array(data)
	plt.plot(x, y)
	plt.show()
	
def plots(errs, accs):
	x = np.arange(0, len(accs))
	accs = np.array(accs)
	errs = np.array(errs)

	plt.plot(x, errs, label='err', color='red')
	plt.legend(loc='center right')
	plt.twinx()
	plt.plot(x, accs, label='acc', color='green')
	plt.legend(loc='center right')
	
	plt.xlabel('step')
	plt.title('accuracy and error')
	plt.show()
	
	
def uniq_id(group='default'):
	return Util.uniq_id(group)
	
	
def array_index_select(arr, dim, i):
	"""
	对arr选取dim维度的i索引
	"""
	idx = [slice(None)] * arr.ndim
	idx[dim] = i
	return arr[tuple(idx)]
	
def array_index_plus(arr, dim, i, plus):
	idx = [slice(None)] * arr.ndim
	idx[dim] = i
	arr[tuple(idx)] += plus

def expand_data(data, dim, copies):
	"""
	维度扩展(矩阵维度增大)
	"""
	data_dims_size = len(data.shape)
	new_data_orders = list(range(0, data_dims_size))
	new_data_orders.insert(dim, data_dims_size)

	new_shape = list(data.shape) + [copies]
	return data.repeat(copies).reshape(new_shape).transpose(new_data_orders)

def softmax(data):
	data_max = np.max(data, axis=-1, keepdims=True)
	data_exp = np.exp(data - data_max) # 防止指数计算溢出
	data_sum = np.sum(data_exp, axis=-1, keepdims=True)
	return data_exp / data_sum

if __name__ == '__main__':
	x = np.random.randn(100, 20)
	y = np.random.randn(100, 10)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.1)
	assert x_train.shape == (90, 20)
	assert x_test.shape == (10, 20)
	assert y_train.shape == (90, 10)
	assert y_test.shape == (10, 10)