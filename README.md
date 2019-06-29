## TinyDeepLearning

TinyDeepLearning是一个使用Python从底层实现的简易深度学习框架。API风格模仿Keras，底层使用Tensor自动进行梯度反向传播，计算模型实现了序列模型和计算图模型，其中计算图模型可以实现任意复杂的图计算(包括残差连接、共享参数等)。 该框架主要用于探究深度学习框架的内部实现原理。



## Class

- Model
  - Sequential 
  - Model 
- Layers
  - Activation(relu sigmoid softmax)
  - Faltten
  - Dense
  - Conv2d
  - MaxPooling
  - RNN
  - Dropout
  - Reshape
  - Add
- Loss
  - MSE
  - SoftmaxCrossEntropy
- Optimizer
  - SGD
  - Momentum
  - RMSprop
  - Adam



## 示例

### 序列模型 示例1

```python
from sklearn import datasets
from util import *
from layer import *
from sequential import *
from model import *
import numpy as np

data = datasets.load_digits()
X = data.data
y = data.target
Y = to_categorical(y).astype('int')

model = Sequential(SGD(learning_rate=0.01), SoftmaxCrossEntropy())
model.add(Dense(512, X.shape[-1]))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(10))

errs, accs = model.fit(X, Y, 10)
plots(errs, accs)
```



### 序列模型 示例2

```python
from sklearn import datasets
from util import *
from layer import *
from sequential import *
from model import *
import numpy as np

data = datasets.load_digits()
X = data.data
y = data.target
Y = to_categorical(y).astype('int')
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
```






### 图模型 示例1

```python
from sklearn import datasets
from util import *
from layer import *
from sequential import *
from model import *
import numpy as np


data = datasets.load_digits()
X = data.data
y = data.target
Y = to_categorical(y).astype('int')
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
```



### 图模型 示例2

```python
from sklearn import datasets
from util import *
from layer import *
from sequential import *
from model import *
import numpy as np

data = datasets.load_digits()
X = data.data
y = data.target
Y = to_categorical(y).astype('int')

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
```



### 图模型 示例3

```python
from sklearn import datasets
from util import *
from layer import *
from sequential import *
from model import *
import numpy as np


data = datasets.load_digits()
X = data.data
y = data.target
Y = to_categorical(y).astype('int')
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
```
