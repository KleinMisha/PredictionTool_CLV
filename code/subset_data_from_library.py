import numpy as np
import pandas as pd

'''
Get subset from complete data set I wish to use for training
'''

def get_homology(row):
    g = np.array(list(str(row['guide'])))
    t = np.array(list(str(row['target'])))
    return g == t


'''
setttings
'''

# subset data
guide_name = 'CLTA1'
concentration = '100nM'
PAM_seq = 'NGG'
MM_count = [1,2]


filename = "Pattanayak_" + guide_name + "_" + concentration + "_" + PAM_seq
for mm_count in MM_count:
    filename += '_'+str(mm_count) + 'MM'
filename += "_sequences.txt"


Pattanayak = pd.read_excel('../data/Pattanayak_Liu_NBT_2673_Data.xlsx')
training = Pattanayak[(Pattanayak.Concentration == concentration)
                      & (Pattanayak.PAM == PAM_seq)
                      & (Pattanayak.name == guide_name)
                      & ( (Pattanayak['target - MMcount'] == MM_count[0]) | (Pattanayak['target - MMcount'] == MM_count[1]) ) ]
training_sequences = []
for i in range(len(training)):
    row = training.iloc[i]
    training_sequences.append(get_homology(row)[:-2])

np.savetxt(filename, training_sequences)