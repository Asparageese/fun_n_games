import tensorflow as tf 
import numpy as np

x,y = np.load("BS_x.npy",allow_pickle=True), np.load("BS_y.npy",allow_pickle=True)
d_mod = np.shape(x[0])

def loss_function(y,y_pred):
	return abs(y - y_pred)

class qmod(tf.Module):
	def __init__(self):
		self.a = tf.Variable(tf.random.uniform(d_mod))
		self.b = tf.Variable(tf.random.uniform(d_mod))
		self.c = tf.Variable(tf.random.uniform(d_mod))

	def __call__(self,x):
		x = tf.square(self.a*x) + self.b*x + self.c
		return tf.nn.softmax(x)

model = qmod()

samples = 2000
epochs = 50


for epoch in range(epochs):
	with tf.GradientTape() as tape:
		loss_value = loss_function(y,model(x))
	grads = tape.gradient(loss_value,model.variables)
	for g,v in zip(grads,model.variables):
		v.assign_sub(g)
	loss = loss_function(y,model(x))
	if epoch % 10 == 0:
		print("current_loss:",loss.numpy())




