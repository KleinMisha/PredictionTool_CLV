import numpy as np
import CRISPR_Delta_breakpoints_slopes as CRISPR
import sys
import SA_move_subset as SA
import functools


def main(argv):
    sequences_file = argv[1]
    scores_file = argv[2]
    fit_results_file = argv[3]
    monitor_file = argv[4]

    # INPUT number of distances between breakpoints there are in both C and I
    nmbr_params_C = int(argv[5])
    nmbr_params_I = int(argv[6])


    '''
    Load data
    (check out: 'subset_data_from_library.py')
    '''
    training_sequences = np.loadtxt(sequences_file)
    training_values = np.loadtxt(scores_file)
    no_chi_squared = np.ones(len(training_values))

    '''
    Prepare Starting Guess
    
    input will be number of breakpoints for both matches and mismatches. Or rather number of distances between them.
    Construct the staring guess and upper/lower bounds by infering their sizes from input values. 
    '''
    # Starting configuration of breakpoints:
    # Have breakpoints start in configuration straight after each other:
    breakpointsC = [i for i in range(1,nmbr_params_C+1)]
    breakpointsI = [i for i in range(1,nmbr_params_I+1)]
    breakpoints =  breakpointsC + breakpointsI

    # Starting guess for slopes and appropiate upper and lower bounds:
    slopes = [5.0] * ((nmbr_params_C+3)+(nmbr_params_I+1))
    upper_bound_slopes = [10.0]* ((nmbr_params_C+3)+(nmbr_params_I+1))
    lower_bound_slopes = [-10.0] + [0.1]*(nmbr_params_C+1) + [-10.0] + [0.1]*(nmbr_params_I+1)

    starting_guess = breakpoints + slopes
    upper_bound = upper_bound_slopes
    lower_bound = lower_bound_slopes

    '''
    model to fit
    '''
    my_model = functools.partial(CRISPR.Pclv, nmbr_params_C=nmbr_params_C,nmbr_params_I=nmbr_params_I)

    '''
    simulated annealing fit
    '''

    fit_result = SA.sim_anneal_fit(model=my_model,
                                   xdata=training_sequences,
                                   ydata=np.array(training_values),
                                   yerr=no_chi_squared,
                                   Xstart=np.array(starting_guess),
                                   lwrbnd=np.array(lower_bound),
                                   upbnd=np.array(upper_bound),
                                   use_multiprocessing=False,
                                   nprocs=1,
                                   Tfinal=0.,
                                   tol=0.001,
                                   Tstart=100,
                                   N_int=1000,
                                   cooling_rate=0.95,
                                   use_relative_steps=False,
                                   delta_breakpoints=0.1,
                                   delta_slopes=2.0,
                                   output_file_results=fit_results_file,
                                   output_file_monitor=monitor_file,
                                   objective_function='chi_squared',
                                   nmbr_breakpoints_C=nmbr_params_C,
                                   nmbr_breakpoints_I=nmbr_params_I
                                   )

    print("input parameter set: ", starting_guess)
    print("fitted parameter set: ", fit_result)
    return



if __name__ == "__main__":
    main(sys.argv)