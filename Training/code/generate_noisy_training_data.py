import numpy as np
import CRISPR_TargetRecognition as CRISPR
import pandas as pd


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

# test parameter set:
DeltaPAM = 0.0
DeltaC = 0.5
DeltaI = 4.0
DeltaCLV = -10.0
Delta = [DeltaPAM, DeltaC, DeltaI, DeltaCLV]

# noisy data (sigma of Gaussian):
noise_amplitude = 0

'''
Get subset from complete data set I wish to use for training
'''
filename = "Pattanayak_" + guide_name + "_" + concentration + "_" + PAM_seq + "_sequences.txt"

Pattanayak = pd.read_excel('../data/Pattanayak_Liu_NBT_2673_Data.xlsx')
training = Pattanayak[(Pattanayak.Concentration == concentration)
                      & (Pattanayak.PAM == PAM_seq)
                      & (Pattanayak.name == guide_name)]
training_sequences = []
for i in range(len(training)):
    row = training.iloc[i]
    training_sequences.append(get_homology(row)[:-2])

np.savetxt(filename, training_sequences)

'''
Calculate mock data based on given parameter set to test the fitting procedure
'''

noise = np.random.normal(loc=0.0, scale=noise_amplitude, size=len(training))

on_target_pclv = CRISPR.Pclv([np.ones(20)], Delta, model='minimal_model')



# Ensure that the data stil produces a positve value. This is to mimic the experimental data
training_data = np.maximum(CRISPR.Pclv(training_sequences, Delta, model='minimal_model') / on_target_pclv + noise, 0)

filename = 'generated_pclv_noise_' + str(noise_amplitude) + '.txt'
np.savetxt(filename, training_data)




