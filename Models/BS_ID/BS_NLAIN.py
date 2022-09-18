import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Attention, LayerNormalization

x = np.load('BS_x.npy',allow_pickle=True)
y = np.load('BS_y.npy',allow_pickle=True)

x,y = np.array(x,np.float32), np.array(y,np.float32)

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




class gt_model(tf.keras.Model):
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

d_mod = 128
rate = 0.3


input_layer = Input(shape=(x.shape[1:]))
e1 = encoder(rate=0.2,dff=(d_mod*2)+2,d_mod=d_mod)(input_layer)
y_pred = Dense(2,activation='relu')(e1)

model = gt_model(input_layer,y_pred)
model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=2)


model.compile(optimizer="Nadam", loss="MeanAbsoluteError", metrics=["accuracy"])
with tf.device('/cpu:0'):
	model.fit(x, y, epochs=7, batch_size=6, validation_split=0.1,shuffle=True,use_multiprocessing=True,callbacks=[callback])