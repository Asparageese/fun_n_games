import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, MultiHeadAttention, Dropout, BatchNormalization, Attention, LayerNormalization


def pointwise_ff(dff,d_mod):
	return tf.keras.Sequential([
		Dense(dff,activation='relu'),
		Dense(d_mod,activation='relu')
		])

class encoder(tf.keras.layers.Layer):
	def __init__(self, d_mod, dff, rate):
		super(encoder,self).__init__()
		self.d_mod = d_mod
		self.dff = dff
		self.rate = rate

		self.a1 = Attention()
		self.pwl = pointwise_ff(dff=self.dff,d_mod=self.d_mod)

		self.d1 = Dropout(self.rate)
		self.lnorm = LayerNormalization()
		self.bnorm = BatchNormalization()

	def call(self, inputs):
		x = self.a1([inputs,inputs,inputs])
		x = self.d1(x)
		x = self.bnorm(x)
		x = self.pwl(x)
		return self.lnorm(x)




class encoder_decoder(tf.keras.Model):
	@tf.function
	def train_step(self,data):
		x,y = data

		with tf.GradientTape() as tape:
			y_pred = self(x,training=True)# preform predictions
			loss = self.compiled_loss(y,y_pred,regularization_losses=self.losses) # compute difference between true and predicted values

		trainable_vars = self.trainable_variables # establish all trainable variables
		gradients = tape.gradient(loss,trainable_vars) # preform autograd to calculate adjusted new state of variables

		self.optimizer.apply_gradients(zip(gradients,trainable_vars)) # apply change calculated
		self.compiled_metrics.update_state(y,y_pred) # calculate new metrics and update
		return {m.name: m.result() for m in self.metrics}

# test data 
d_mod = 128
rate = 0.3

dataset_size = 250000
x, y = np.random.rand(dataset_size), np.random.rand(dataset_size)

input_layer = Input(shape=(1))
e_test = encoder(d_mod=d_mod,dff=(d_mod*2)+2,rate=rate)(input_layer)
y_pred = Dense(1,activation='relu')(e_test)

model = encoder_decoder(input_layer,y_pred)
model.summary()

with tf.device('/cpu:0'):
	model.compile(loss=tf.keras.losses.KLDivergence(),optimizer=tf.keras.optimizers.Adam(),metrics=['mse','accuracy'])
	model.fit(x,y,epochs=10,batch_size=24)