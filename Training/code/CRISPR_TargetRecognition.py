import numpy as np

def Pclv(sequences, Delta,model='minimal_model'):
    '''
    Calculates the probability to cleave according to kinetic model.
    See 'Hybridisation Kinetics Explains Targeting Rules' for more info.

    Can be used together with SimulatedAnnealing.py

    For now it will call the 'minimal model'

    :param sequences: encoded guide-target pairs. For minimal model this is "guide==target"
    :param Delta: list or array of Kinetic Biases
    :param model: name of the parameterization used for the kinetic biases.
    :return: Pclv for every member in the list of sequences
    '''

    pclv = []
    for seq in sequences:

        # get entire transition landscape:
        DeltaT = TransitionLandscape(model,Delta, seq)

        # For Pclv, I need the sum of the exponents of DeltaT_n
        exp_of_T = np.sum(np.exp(-DeltaT))
        pclv.append(1.0 / (1.0 + exp_of_T))
    return np.array(pclv)


def TransitionLandscape(model, Delta, sequence):
    '''
    Calculates the Transition Landscape
    :param model: name of the parameterization used for the Kinetic Biases
    :param Delta: list or array of Kinetic Biases, numerical values
    :param sequence: code for the guide-to-target hybrid
    :return: An array with all the cumulative kinetic biases as a function of targeting progression
    '''


    DeltaT = []
    if model == 'minimal_model':
        DeltaPAM = Delta[0]
        DeltaC = Delta[1]
        DeltaI = Delta[2]
        DeltaCLV = Delta[3]
        for n in range(1, len(sequence) + 1):
            nC = np.sum(sequence[0:n])
            DeltaT_n = DeltaPAM + nC * DeltaC - (n - nC) * DeltaI - KroneckerDelta(n, len(sequence)) * DeltaCLV
            DeltaT.append(DeltaT_n)
    return np.array(DeltaT)


def KroneckerDelta(n, k):
    '''
    The Kronecker Delta funtion makes it easier to write down the transition landscape
    '''
    if n == k:
        return 1.0
    else:
        return 0.0