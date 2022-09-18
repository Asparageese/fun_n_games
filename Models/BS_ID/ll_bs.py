import tensorflow as tf 
import numpy as np

x,y = np.load('BS_x.npy',allow_pickle=True), np.load('BS_y.npy',allow_pickle=True)

d_dim = np.shape(x[0])
print("x shape",d_dim)

def loss(y,y_pred):
	return tf.reduce_mean(tf.square(y_pred - y))

class twostage(tf.Module):
	def __init__(self):
		self.k1 = tf.Variable(tf.random.uniform(d_dim))
		self.k2 = tf.Variable(tf.random.uniform(d_dim))

	def __call__(self,x):
		return tf.nn.relu(self.k1 * x + self.k2)

model = twostage()


samples = 10
losses = []
for i in range(samples):
	with tf.GradientTape() as tape:
		loss_value = loss(y[i],model(x))
	grads = tape.gradient(loss_value,model.variables)
	for g,v in zip(grads,model.variables):
		v.assign_sub(g)
	print(loss_value.numpy())