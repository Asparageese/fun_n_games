import tensorflow as tf
import numpy as np 

folders = glob('D:/fng/fun_n_games/DATASETS/KAPD/*')
print(folders)

dataset = []
labels = []
for folder in folders:
	files = glob(folder+'/*')
	print(files)