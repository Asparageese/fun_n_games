from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization, Conv2D, MaxPooling2D,Flatten
import tensorflow as tf
import numpy as np 

x,y = np.load("KAPD_X.npy"), np.load("KAPD_Y.npy")

dff = 128
d_mod = 32
rate = 0.1

va1 = 64
va2 = 64
va3 = 64

def pointwise_ff(dff,d_mod):
	return tf.keras.Sequential([
		Dense(dff,activation='relu'),
		LayerNormalization(),
		Dense(d_mod,activation='softmax'),
		])

class vae(tf.keras.layers.Layer):
	def __init__(self,dff,d_mod,rate,va1,va2,va3):
		super(vae,self).__init__()
		self.dff = dff
		self.d_mod = d_mod
		self.rate = rate
		self.va1 = va1
		self.va2 = va2
		self.va3 = va3


		self.c1 = Conv2D(self.va1,(2, 2),activation='relu')
		self.c2 = Conv2D(self.va2,(2, 2),activation='relu')
		self.c3 = Conv2D(self.va2,(2, 2),activation='relu')
		self.c4 = Conv2D(self.va3,(2, 2),activation='relu')

		self.flat = Flatten()
		self.pwise = pointwise_ff(self.dff,self.d_mod)

		self.pool1 = MaxPooling2D()
		self.pool2 = MaxPooling2D()


		self.d1 = Dropout(rate=self.rate)
		self.d2 = Dropout(rate=self.rate)
		self.d3 = Dropout(rate=self.rate)

		self.n1 = BatchNormalization()
		self.n2 = BatchNormalization()

		self.l1 = LayerNormalization()

	def call(self, inputs):
	
		x = self.c1(inputs)
		x = self.d1(x)
		x = self.n1(x)
		x = self.pool1(x)

		x = self.c2(x)
		x = self.d2(x)
		x = self.pool2(x)
		x = self.c3(x)
		x = self.n2(x)

		x = self.c4(x)
		x = self.d3(x)
		x = self.l1(x)
		x = self.flat(x)
		return self.pwise(x)

class KAPD_NN(tf.keras.Model):
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

input_layer = Input(shape=(x.shape[1:]))
vae_test = vae(dff=dff,d_mod=d_mod,rate=rate,va1=va1,va2=va2,va3=va3)(input_layer)
y_hat = Dense(4,activation='relu')(vae_test)

model =  KAPD_NN(input_layer,y_hat)

model.summary()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2)

with tf.device('/GPU:0'):
	model.fit(x,y,epochs=6,batch_size=1,shuffle=True,use_multiprocessing=True,callbacks=[callback],validation_split=0.1)