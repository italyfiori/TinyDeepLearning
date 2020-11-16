## TinyDeepLearning
TinyDeepLearning is a simple deep learning framework implemented from the scratch using Python. API style imitates Keras, the bottom layer uses the Tensor to carry on the gradient backpropagation automatically. the computational model implements the sequence model and the computational graph model, in which the computational graph model can realize any form of graph calculation (including residual connection, shared parameters, etc). This framework is mainly used to explore the internal implementation principle of deep learning framework.



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



## Examples

### Sequence model example 1

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



### Sequence model example 2

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






### graph model example 1

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



### graph model example 2

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



### graph model example 3

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
