from sklearn import datasets
from util import *
from layer import *
from sequential import *
from model import *
import numpy as np

def test_seq_dense(X, Y):
	model = Sequential(SGD(learning_rate=0.01), SoftmaxCrossEntropy())
	
	model.add( Dense(512, input_shape=X.shape[1:]) )
	model.add(Activation('relu'))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(512))
	model.add(Activation('relu'))
	
	model.add(Dense(10))
	errs, accs = model.fit(X, Y, 10)
	plots(errs, accs)
	
def test_seq_conv(X, Y):
	X = X.reshape((-1, 1, 8, 8))
	model = Sequential(SGD(learning_rate=0.01), SoftmaxCrossEntropy())
	
	model.add(Conv2D(64, filter_shape=(3,3), input_shape=(1,8,8)))
	model.add(Activation('relu'))
	model.add(MaxPooling((2,2), stride=2))
	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dense(10))
	
	errs, accs = model.fit(X, Y, 50)
	plots(errs, accs)

def test_model_dense(X, Y):
	x = Input(X.shape)
	t0 = Dense(512)(x)
	t = Activation('relu')(t0)
	fc = Dense(512) 
	t = Dropout(0.2)(t)
	t = fc(t)
	t = Activation('relu')(t)
	t = Dropout(0.2)(t)
	t = fc(t)
	t = Activation('relu')(t)
	y1 = Dense(10)(t)
	y2 = Dense(10)(t0)
	y = Add()(y1, y2)
	model = Model(x, y)
	model.compile(Adam(learning_rate=0.01), SoftmaxCrossEntropy())
	errs, accs = model.fit(X, Y, 50)
	plots(errs, accs)

def test_model_conv(X, Y):
	X = X.reshape((-1, 1, 8, 8))
	
	x = Input(X.shape)
	t = Conv2D(64, filter_shape=(3,3))(x)
	t = Activation('relu')(t)
	t = MaxPooling((2,2), stride=2)(t)
	t = Flatten()(t)
	t = Dense(256)(t)
	t = Activation('relu')(t)
	y = Dense(10)(t)
	
	model = Model(x, y)
	model.compile(SGD(learning_rate=0.01), SoftmaxCrossEntropy())
	errs, accs = model.fit(X, Y, 50)
	plots(errs, accs)
	
def test_seq_rnn(X, Y):
	X = X.reshape((-1, 8, 8))
	model = Sequential(SGD(learning_rate=0.01), SoftmaxCrossEntropy())
	model.add(RNN(70, input_shape=(8, 8), return_type='sequence'))
	model.add(RNN(50, return_type='last_state'))
	model.add(Dense(64))
#	model.add(Dropout(0.3))
	model.add(Dense(10))
	errs, accs = model.fit(X, Y, 50)
	plots(errs, accs)

def test_model_rnn(X, Y):
	X = X.reshape((-1, 8, 8))
	
	x = Input(X.shape)
	t = RNN(70, return_type='sequence')(x)
	t = Activation('relu')(t) 
	t = RNN(50, return_type='last_state')(t)
	t = Activation('relu')(t)
	t = Dense(256)(t)
	t = Activation('relu')(t)
	y = Dense(10)(t)
	
	model = Model(x, y)
	model.compile(SGD(learning_rate=0.01), SoftmaxCrossEntropy())
	errs, accs = model.fit(X, Y, 100)
	plots(errs, accs)
	
def test_3d_dense(X, Y):
	X = X.reshape((-1, 8, 8))
	
	x = Input(X.shape)
	t = RNN(70, return_type='sequence')(x)
	t = Activation('relu')(t) 
	t = Dense(50)(t)
	t = Activation('relu')(t)
	t = Reshape((-1, 8*50))(t)
	y = Dense(10)(t)
	
	model = Model(x, y)
	model.compile(SGD(learning_rate=0.01), SoftmaxCrossEntropy())
	errs, accs = model.fit(X, Y, 1000)
	plots(errs, accs)

def test_seq_regression():
	a = np.random.randn(100, 2)
	b = 03. * a[:, 0] + 0.9 * a[:, 1] 
	b = b.reshape(len(b), -1) + np.random.rand(100, 1) * 1e-15

	model = Sequential(SGD(), SoftmaxCrossEntropy())
	model.add(Dense(10, input_shape=(2,)))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.fit(a, b, 100)
	
def test_model_regression():
	a = np.random.randn(100, 2)
	b = 100 * a[:, 0] + np.power( a[:, 1], 2) *10
	b = b.reshape(len(b), -1) + np.random.rand(100, 1) * 3
		
	x = Input((None, 2))
	t = Dense(5)(x)
	t = Activation('relu')(t)
	y = Dense(1)(t)
	
	model = Model(x, y)
	model.compile(SGD(learning_rate=0.01), SoftmaxCrossEntropy())
	errs, accs = model.fit(a, b, 10)


if __name__ == '__main__':
	data = datasets.load_digits()
	X = data.data
	y = data.target
	Y = to_categorical(y).astype('int')
	
	test_model_rnn(X, Y)
#	test_model_regression(X, Y)
