import tensorflow as tf 
import numpy as np

class linear(tf.Module):
	def __init__(self, value):
		k1 = tf.Variable(value)
		k2 = tf.Variable([])

	@tf.function
	def multiply(self, x):
		return (x * self.k1) + k2

test = linear(2)
print(test)




