import numpy as np
import pandas as pd
cfd_table = pd.read_excel('../data/CFD_Doench/STable 19 FractionActive_dlfc_lookup.xlsx')

def CCTop(guide, target):
    '''
    CRISPR/Cas9 Target online predictor (CCTop) score as described in Stemmer et al.

    :param guide:
    :param target:
    :return:
    '''

    mm_pos = np.where(guide != target)[0]
    score = 0
    for x in mm_pos:
        score += 1.2 ** (x + 1)
    return max(1, score)


def MITscore(guide, target):
    '''
    the MIT score as described in Hsu et al. and their website

    :param guide:
    :param target:
    :return:
    '''

    M = [0, 0, 0.014, 0, 0, 0.395, 0.317, 0, 0.389, 0.079, 0.445, 0.508, 0.613, 0.851, 0.732, 0.828, 0.615, 0.804,
         0.685, 0.583]
    mm_pos = np.where(guide != target)[0]
    n_mm = len(mm_pos)

    if n_mm > 0:
        dtot = 0
        for pos1 in range(0, n_mm):
            for pos2 in range(pos1 + 1, n_mm):
                dtot += abs(mm_pos[pos1] - mm_pos[pos2])
        if n_mm > 1:
            davg = dtot / (n_mm * (n_mm - 1) / 2.)
        else:
            davg = 0.

        factor_distance = 1 / ((19 - davg) / 19 * 4 + 1) * 1. / n_mm ** 2

        factor_positions = 1
        for pos in mm_pos:
            factor_positions *= (1 - M[pos])

        score = factor_distance * factor_positions
    else:
        score = 1.0
    return score * 100


def CFD(guide, target):
    '''
    Cutting Frequency Determination (CFD) score as described in Doench et al.
    :param guide:
    :param target:
    :param cfd_table:
    :return:
    '''

    DNA = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
    RNA = {0: 'A', 1: 'U', 2: 'C', 3: 'G'}
    mm_pos = np.where(guide != target)[0]
    # table shows the target strand (complementary), we search for homology
    target_strand = 3 - target

    score = 1.
    for MMpos in mm_pos:
        MMtype = 'r' + RNA[int(guide[MMpos])] + ':d' + DNA[int(target_strand[MMpos])]
        CFD = \
        cfd_table[(cfd_table['Mismatch Type'] == MMtype) & (cfd_table['Position'] == MMpos + 1)]['Percent-Active'].iloc[
            0]
        score *= CFD
    return score
