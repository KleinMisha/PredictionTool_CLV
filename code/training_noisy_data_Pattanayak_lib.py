

''''
Initial controls for training of Pclv
- Use library of Pattanayak (Nature Biotech. 2013) as training set (1 guide, 1 PAM sequence, 1 concentration)
    --> see 'generate_noisy_training_data.py'
- Generated noisy data based on 'minimal model' for Pclv and added random value (set noise amplitude)
    --> see 'generate_noisy_training_data.py'
- Sent to hpc05 cluster and use multiprocessing
'''


# enable the import from the file folder
import sys
import numpy as np
# sys.path.append('/Users/mklein1/Documents/PhD_Martin_Depken_Group/My_Modules/GitHub/Simulated_Annealing')
# sys.path.append('./') # current folder

PATH_HPC05 = '/home/mklein1/Training_Pclv/'

# on hpc05 cluster:
sys.path.append(PATH_HPC05)
import SimulatedAnnealing as SA
import CRISPR_TargetRecognition as CRISPR




'''
Load data

(check out: 'generate_noisy_training_data.py')
'''
# subset data
guide_name = 'CLTA1'
concentration = '100nM'
PAM_seq = 'NGG'

filename1 = PATH_HPC05 + "Pattanayak_" + guide_name + "_" + concentration + "_" + PAM_seq + "_sequences.txt"


# noisy data (sigma of Gaussian):
noise_amplitude = 0
filename2 = PATH_HPC05 + 'generated_pclv_noise_' + str(noise_amplitude) + '.txt'

training_sequences = np.loadtxt(filename1)
training_values = np.loadtxt(filename2)
no_chi_squared = np.ones(len(training_values))

'''
Starting Guess
'''

# The values used to generate the data
DeltaPAM = 0.0
DeltaC = 0.5
DeltaI = 4.0
DeltaCLV = -10.0

starting_guess= [5.0, 3.0, 1.0, -4.0]
upper_bound  = [np.inf, np.inf, np.inf, np.inf ]
lower_bound  = [0.0, 0.0, 0.0, -np.inf]

'''
simulated annealing fit
'''

fit_result = SA.sim_anneal_fit(model=CRISPR.Pclv,
             xdata=training_sequences,
             ydata=training_values,
             yerr=no_chi_squared,
             Xstart= starting_guess,
             lwrbnd=lower_bound,
             upbnd=upper_bound,
             use_multiprocessing=True,
             nprocs=8,
             Tfinal=0.5,
             tol=0.001,
             Tstart=10,
             N_int = 1000,
             cooling_rate= 0.95,
             output_file_results='/home/mklein1/Training_Pclv/fit_4_1_2017_noise_%s.txt' %(noise_amplitude),
             output_file_monitor='/home/mklein1/Training_Pclv/monitor_4_1_2017_noise_%s.txt' %(noise_amplitude)
             )


print "input parameter set: ", starting_guess
print "fitted parameter set: ", fit_result










