import numpy as np 
from glob import glob 
from PIL import Image
from tqdm import tqdm
from sklearn.utils import shuffle
from skimage.transform import resize

folders = glob('D:/fng/fun_n_games/DATASETS/Micro_Organism/*')
print(folders)

compact_set = []
mag_index = []
for i in folders:
	files = glob(i+'/*')
	mag_index.append(len(files))
	for image_path in tqdm(files):
		image_data = Image.open(image_path).convert("L")
		compact_set.append(resize(np.array(image_data),(255,255)))


compact_set = np.array(compact_set)
print(np.shape(compact_set))

labels = []
for i in range(len(mag_index)):
	for a in range(mag_index[i]):
		labels.append(i)

compact_set, labels = shuffle(compact_set, labels, random_state=0)
np.save('x_data.npy',compact_set)
np.save('labels.npy',labels)



