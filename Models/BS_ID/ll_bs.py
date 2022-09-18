import tensorflow as tf 
import numpy as np

sample = 1000

y = []
x = []
tile = np.arange(0,10,1)
for i in range(sample):
	x.append(tile)
	y.append(np.random.rand(10))

def loss(target_y, predicted_y):
  return tf.reduce_mean(tf.square(target_y - predicted_y))

class linear(tf.Module):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.w = tf.Variable(tf.random.uniform([10]),name='w')
		self.b = tf.Variable(tf.random.uniform([10]),name='b')

	def __call__(self,x):
		return self.w * x + self.b

test = linear()

for i in range(sample):
	s = y[i]
	with tf.GradientTape() as tape:
		y_loss = loss(y,test(x[i]))

	grads = tape.gradient(y_loss,test.variables)