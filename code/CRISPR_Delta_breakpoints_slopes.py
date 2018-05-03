##############################################################
# Systematically increasing the complexity of parameteririsation
# 1. Have continuous functions f_C(x) and f_I(x) that are peicewise linear
# 2. Number of 'break points (changes of slopes)' and value of slopes determine number of unique Delta values
# 3. Delta = f(x+1) - f(x) ( np.diff() )
#############################################################
import numpy as np
import CRISPR_TargetRecognition


def Pclv(sequences,parameters,nmbr_params_C=0,nmbr_params_I=0):
    '''
    First translates set of breakpoints and slopes via a peicewise linear function into
    (transition state-)energy values Delta

    Afterwards uses Delta to calculate Pclv

    :param sequences: binary encoded sequence: 1 for match and 0 for mismatch
    :param parameters: all parameters concatenated (needed for fitting using SA) -->[distances, slopes]
    :param nmbr_params_C: number of variable distances between internal breakpoints in ON-target landscape
    :param nmbr_params_I: number of variable distances between breakpoints in mimsatch_penalty function
    :return: Pclv for every member of sequences
    '''
    breakpoints = parameters[:(nmbr_params_C+nmbr_params_I)]
    slopes = parameters[(nmbr_params_C+nmbr_params_I):]
    Delta = get_delta_genral(breakpoints,slopes,nmbr_params_C)
    return CRISPR_TargetRecognition.Pclv(sequences,Delta,model='general_model')

def compare_to_on_target(sequences,parameters,nmbr_params_C=0,nmbr_params_I=0):
    '''
    First translates set of breakpoints and slopes via a peicewise linear function into
    (transition state-)energy values Delta

    Afterwards uses Delta to calculate Pclv

    :param sequences: binary encoded sequence: 1 for match and 0 for mismatch
    :param parameters: all parameters concatenated (needed for fitting using SA) -->[distances, slopes]
    :param nmbr_params_C: number of variable distances between internal breakpoints in ON-target landscape
    :param nmbr_params_I: number of variable distances between breakpoints in mimsatch_penalty function
    :return: Pclv for every member of sequences
    '''
    breakpoints = parameters[:(nmbr_params_C+nmbr_params_I)]
    slopes = parameters[(nmbr_params_C+nmbr_params_I):]
    Delta = get_delta_genral(breakpoints,slopes,nmbr_params_C)
    return CRISPR_TargetRecognition.compare_to_on_target(sequences,Delta,model='general_model')

def single_mismatches(parameters,nmbr_params_C,nmbr_params_I):
    distances = parameters[:(nmbr_params_C+nmbr_params_I)]
    slopes = parameters[(nmbr_params_C+nmbr_params_I):]
    Delta = get_delta_genral(distances,slopes,nmbr_params_C)
    return CRISPR_TargetRecognition.single_mismatches(Delta, guide_length= 20,model='general_model',normed=True)



def get_delta_genral(breakpoints,slopes,NmbrC):
    '''
    Convert parameters of continuos function to descrete set of Delta-values
    :param Nmbr_C: number of (internal) distances between breakpoints in DeltaC (range from 0 to 19)
    :param Nmbr_I: number of distances between breakpoints in DeltaI (range from 0 to 20)
    :return:
    '''

    DeltaC = get_delta_match(breakpoints[:NmbrC], slopes[:(NmbrC+3)])
    DeltaI = get_delta_mismatch(breakpoints[NmbrC:],slopes[(NmbrC+3):])
    Delta = np.concatenate((DeltaC,DeltaI))
    return Delta


''''
Additional functions 
'''

def get_delta_mismatch(breakpoints,slopes,Nguide=20):
    function_I, _ = function_mismatch(breakpoints,slopes,Nguide)
    return np.diff(function_I)


def get_delta_match(breakpoints,slopes,Nguide=20):
    function_C,_ = function_match(breakpoints,slopes,Nguide)
    return np.diff(function_C)

def function_match(breakpoints, slopes, Nguide=20):
    # /* Let n be index  [-1,0,1,2,....,21]  (states including solution, PAM and post-cleavage)*\:
    n = np.arange(0, Nguide + 1)

    # /* construct 'pivot points'/'break points' in landscape *\:
    breakpoints = np.concatenate(([-1,0],breakpoints,[Nguide-1,Nguide]))

    # /* use break points and provided slopes to evaluate peicewise linear function *\:
    lines = np.array([0])
    lines = np.concatenate((lines, np.cumsum(np.diff(breakpoints) * slopes)))
    y = [0]
    for nval in n:
        # 1) find last breakpoint you have passed:
        index = np.where(nval > breakpoints)[0][-1]
        # 2) Add slopes up until that point
        # 3) Compensate if breakpoint is between two integer values:
        y.append(lines[index] + slopes[index] * (nval - breakpoints[index]))
    return np.array(y), breakpoints


def function_mismatch(breakpoints, slopes, Nguide=20):
    # /* Let n be index  [0,1,2,....,21]  (solution state not needed: only use canonical PAM) *\:
    n = np.arange(1, Nguide + 1)

    # /* construct 'pivot points'/'break points' in landscape *\:
    # 2) I provide breakpoints:
    breakpoints = np.concatenate(([0],breakpoints,[Nguide]))

    # /* use break points and provided slopes to evaluate peicewise linear function *\:
    lines = np.array([0])
    lines = np.concatenate((lines, np.cumsum(np.diff(breakpoints) * slopes)))
    y = [0]
    for nval in n:
        # 1) find last breakpoint you have passed:
        index = np.where(nval > breakpoints)[0][-1]
        # 2) Add slopes up until that point
        # 3) Compensate if breakpoint is between two integer values:
        y.append(lines[index] + slopes[index] * (nval - breakpoints[index]))
    return np.array(y), breakpoints



