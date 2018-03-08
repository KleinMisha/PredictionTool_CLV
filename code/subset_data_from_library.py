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


file_seq = "Pattanayak_" + guide_name + "_" + concentration + "_" + PAM_seq
for mm_count in MM_count:
    file_seq += '_'+str(mm_count) + 'MM'


file_score = file_seq
file_seq += "_sequences.txt"
file_score += "_scores.txt"


Pattanayak = pd.read_excel('../data/Pattanayak_Liu_NBT_2673_Data.xlsx')
training = Pattanayak[(Pattanayak.Concentration == concentration)
                      & (Pattanayak.PAM == PAM_seq)
                      & (Pattanayak.name == guide_name)
                      & ( (Pattanayak['target - MMcount'] == MM_count[0]) | (Pattanayak['target - MMcount'] == MM_count[1]) ) ]
training_sequences = []
for i in range(len(training)):
    row = training.iloc[i]
    training_sequences.append(get_homology(row)[:-2])


training_scores = training['Score']

np.savetxt(file_seq, training_sequences)
np.savetxt(file_score, training_scores)