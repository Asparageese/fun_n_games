import pandas as pd
import numpy as np
from tqdm import tqdm 
from sklearn.utils import shuffle
from sklearn.preprocessing import scale

dataset = pd.read_csv('D:/fng/fun_n_games/DATASETS/brain_stroke.csv').to_numpy()

## format ##########  gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke
# 0,4,5,6,9
identifiers = np.array([[['Male'],['Female']],[['Yes'],['No']],[['Private'],['Self-empolyed'],['Govt_job']],[['Urban'],['Rural']],[['fomerly smoked'],['never smoked'],['smokes']]])
print(np.shape(identifiers))

def preprocessing(array,i_d,id_num,alteration):
	array = array[:,alteration]
	for v in array:
		if i_d == v:

