import numpy as np 
from glob import glob 
from PIL import Image
from sklearn.utils import shuffle
from skimage.transform import resize
from tqdm import tqdm

folders = glob('D:/fng/fun_n_games/DATASETS/KAPD/*')
print(folders)

dataset = []
l_magnitudes = []
for folder in folders:
	files = glob(folder+'/*')
	l_magnitudes.append(np.size(files))
	for sample in tqdm(files):
		x = Image.open(sample)
		dataset.append(resize(np.array(x),(255,255,3)))

dataset=np.array(dataset)
print(np.shape(dataset))

labels = []
c = 0
for i in l_magnitudes:
	for l in range(i):
		labels.append(c)
	c = c+1

x,y = shuffle(dataset,labels,random_state=0)
np.save("KAPD_X",x)
np.save("KAPD_Y",y)