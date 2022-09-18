import pandas as pd
import numpy as np
from tqdm import tqdm 
from sklearn.utils import shuffle
from sklearn.preprocessing import scale

def feature_alterations(array,feature,id_classes):
    array = array[:,feature]
    processed_data = []
    for i in range(np.size(array)):
        c = 0.
        for i_d in id_classes:
            if array[i] == i_d:
                processed_data.append(c)
            c = c + 1
    return np.array(processed_data)

dataset = pd.read_csv('D:/fng/fun_n_games/DATASETS/brain_stroke.csv').to_numpy()

## format ##########  gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke
# 0,4,5,6,9

ida = np.array([['Male'],['Female']])
idb = np.array(['Yes','No'])
idc = np.array(['Private','Self-employed','Govt_job','children'])
idd = np.array(['Urban','Rural'])
ide = np.array(['smokes','formerly smoked','Unknown','never smoked'])
collective_identities = [ida,idb,idc,idd,ide]

alternations = [0,4,5,6,9]
for i in range(np.size(collective_identities)):
    dataset[:,alternations[i]] = feature_alterations(dataset, alternations[i], collective_identities[i])

x,y = shuffle(dataset[:,:(np.size(dataset[0,:])-1)],dataset[:,-1:],random_state=0)

np.save('BS_x',x)
np.save('BS_y',y)

    
