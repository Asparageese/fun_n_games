import numpy as np 
import tensorflow as tf 
from tensorflow.keras.layers import Input,Flatten,Dense,Conv2D,BatchNormalization,LayerNormalization,Dropout,MaxPooling2D

x = np.load('x_data.npy')
y = np.load('labels.npy')


def pointwise_ff(dff,d_mod):
	return tf.keras.Sequential([
		Dense(dff,activation='relu'),
		Dense(d_mod,activation='relu')
		],name='pointwise_ff')


class VAE(tf.keras.layers.Layer):
	def __init__(self,
		d_mod,
		dff,
		rate):
		super(VAE,self).__init__()
		self.d_mod = d_mod
		self.dff = dff
		self.rate = rate

		self.encoder = tf.keras.Sequential([
			Conv2D(64,(2, 2),activation='relu'),
			Dense(128,activation='relu'),
			Dropout(self.rate),
			BatchNormalization(),
			MaxPooling2D((6,6)),
			Conv2D(32,(2,2),activation='relu'),
			Dense(64,activation='relu'),
			Dropout(self.rate),
			BatchNormalization(),
			MaxPooling2D((5,5)),
			Conv2D(64,(2,2),activation='relu'),
			Dense(128,activation='relu'),
			Dropout(self.rate),
			MaxPooling2D((6,6)),
			LayerNormalization(),
			pointwise_ff(dff=self.dff,d_mod=self.d_mod)
			],name='encoder_mechanisim')

	def call(self,inputs):
		return self.encoder(inputs)
		

class VAE_MODEL(tf.keras.Model):
	@tf.function
	def train_step(self,data):
		x,y = data
		with tf.GradientTape() as tape:
			y_pred = self(x,training=True)
			loss = self.compiled_loss(y,y_pred,regularization_losses=self.losses)

		trainable_vars = self.trainable_variables # establish all trainable variables
		gradients = tape.gradient(loss,trainable_vars) # preform autograd to calculate adjusted new state of variables

		self.optimizer.apply_gradients(zip(gradients,trainable_vars)) # apply change calculated
		self.compiled_metrics.update_state(y,y_pred) # calculate new metrics and update
		return {m.name: m.result() for m in self.metrics}


inputs = Input(shape=(255,255,1))
VAE_layer = VAE(128,256,0.2)(inputs)
flatten = Flatten()(VAE_layer)
y_hat = Dense(8,activation='softmax')(flatten)

model = VAE_MODEL(inputs,y_hat)
model.summary()

with tf.device('/GPU:0'):
	model.compile(optimizer="nadam", loss="SparseCategoricalCrossentropy", metrics=["accuracy"])
	model.fit(x[:400],y[:400],epochs=3,batch_size=2,use_multiprocessing=True,shuffle=True)
	model.evaluate(x[400:],y[400:])