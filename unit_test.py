import unittest

from tensor import *

class TestTensor(unittest.TestCase):
    def test_select(self):
        data = np.array(range(0, 9)).reshape(3,3).astype('int')
        a = Tensor(data, autograd=True)
        b = a.select_index(0, 1)
        c = a.select_index(1, 1)
        d = b + c
        grad = Tensor(np.ones(b.data.shape))
        d.backward(grad)
        self.assertTrue( np.all(a.grad.data == np.array([[0,1,0],[1,2,1],[0,1,0]])) )
        
    def test_add(self):
        a = Tensor([1,2,3,4,5], autograd=True)
        b = Tensor([2,2,2,2,2], autograd=True)
        c = Tensor([5,4,3,2,1], autograd=True)

        d = a + b
        e = b + c 
        f = d + e
        f.backward(Tensor(np.array([1,1,1,1,1])))
        self.assertTrue( np.all( b.grad.data == np.array([2,2,2,2,2])) )
        
    def test_sub(self):
        a = Tensor([1,2,3,4,5], autograd=True)
        b = Tensor([2,2,2,2,2], autograd=True)
        c = a - b
        c.backward(Tensor(np.array([1,1,1,1,1])))
        
        self.assertTrue( np.all( a.grad.data == np.array([1,1,1,1,1])) )
        self.assertTrue( np.all( b.grad.data == np.array([-1,-1,-1,-1,-1])) )

    def test_neg(self):
        a = Tensor([1,2,3,4,5], autograd=True)
        b = -a        
        b.backward(Tensor(np.array([1,1,1,1,1])))
        
        self.assertTrue( np.all( a.grad.data == np.array([-1,-1,-1,-1,-1])) )
        
    def test_mul(self):
        a = Tensor(np.random.randn(3, 4), autograd=True)
        b = Tensor(np.random.randn(3, 4), autograd=True)
        c = a * b
        c.backward(Tensor(np.ones(c.data.shape)))
        
        self.assertTrue( np.all(a.grad.data == b.data) )
        self.assertTrue( np.all(b.grad.data == a.data) )
        
    def test_reshape(self):
        a = Tensor(np.random.randn(3, 4), autograd=True)
        b = a.reshape((4,3))
        b.backward(Tensor(np.ones(b.data.shape)))
        
        self.assertTrue( np.all(a.grad.data == b.grad.data.reshape((3,4)) ) )
        
    def test_refresh(self):
        a = Tensor(np.ones((3,1)), autograd=True)
        b = -a
        c = b + b
        d = b + b
        d.backward(Tensor(np.ones(d.data.shape)))
        self.assertIsNone(a.grad)
        
        a = Tensor(np.ones((3,1)), autograd=True)
        b = -a
        c = b + b
        d = b + b
        d.refresh_dependencies()
        d.backward(Tensor(np.ones(d.data.shape)))
        self.assertIsNotNone(a.grad.data)
        
    def test_concatenate(self):
        a = Tensor(np.ones((3,1)), autograd=True)
        b = Tensor(np.ones((3,2))*2, autograd=True)
        c = Tensor(np.ones((3,2))*2, autograd=True)
        
        d = Tensor.concatenate([a, b, c], axis=1)
        self.assertTrue(np.all(d.data   ))
        
        d.backward(Tensor(np.array(range(15)).reshape(3,5)))
        
        self.assertTrue(np.all( a.grad.data == d.grad.data[:, 0:1] ))
        self.assertTrue(np.all( b.grad.data == d.grad.data[:, 1:3] ))
        self.assertTrue(np.all( c.grad.data == d.grad.data[:, 3:5] ))
        
    def test_max(self):
        data = np.array(range(12)).reshape(3,4)
        a = Tensor(data, autograd=True)

        b = a.max()
        self.assertTrue( np.all( b.data == data.max(axis=1) ) )
        
        grad = Tensor( np.array(range(1,4)).reshape(3,) )      
        
    def test_power(self):
        data = np.array(range(12)).reshape(3,4)
        a = Tensor(data, autograd=True)
        
        num = 2
        b = a.power(num)
        self.assertTrue( np.all( b.data == np.power(a.data, num) ) )
        
        grad_data = np.ones_like(b.data)
        b.backward(Tensor(grad_data))
        self.assertTrue( np.all(a.grad.data == num * np.power(a.data, num-1) * grad_data  ))
        
        
if __name__ == '__main__':
    unittest.main(verbosity=2)