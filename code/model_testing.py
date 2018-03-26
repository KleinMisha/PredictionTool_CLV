import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

def ROC_curve(model_predictions, true_classifications, make_plot=True):
    '''
    make the ROC curve and plot (optional)

    :param model_predictions: continuous output from candidate classifier
    :param true_classifications: binary data from experiments (positives and negatives) you want to predict
    :param make_plot: set to 'TRUE' to plot result
    :return: False Positive Rate(FPR), True Positive Rate(TPR), both as vector, Area Under the Curve (AUC)
    '''
    threshhold = np.array([i * 0.01 for i in range(101)])
    TPR = [1]
    FPR = [1]
    for sigma in threshhold:
        predicted_positives = model_predictions >= sigma

        TP = np.sum((predicted_positives == True) & (true_classifications == True))
        FP = np.sum((predicted_positives == True) & (true_classifications == False))
        TN = np.sum((predicted_positives == False) & (true_classifications == False))
        FN = np.sum((predicted_positives == False) & (true_classifications == True))

        TPR.append(float(TP) / (TP + FN))
        FPR.append(float(FP) / (FP + TN))

    TPR.append(0)
    FPR.append(0)
    if make_plot:
        plt.plot(FPR, TPR)
        plt.xlabel('FPR', fontsize=15);
        plt.ylabel('TPR', fontsize=15);
        plt.title('ROC-curve', fontsize=15);
        sns.despine();

    AUC = np.abs(np.trapz(y=TPR, x=FPR));
    return FPR, TPR, AUC


def PR_curve(model_predictions, true_classifications, make_plot=True):
    '''
    make the PR curve and plot (optional)

    :param model_predictions: continuous output from candidate classifier
    :param true_classifications: binary data from experiments (positives and negatives) you want to predict
    :param make_plot: set to 'TRUE' to plot result
    :return: Precision, Recall, both as vector, Area Under the Curve (AUC)
    '''
    threshhold = np.array([i * 0.01 for i in range(100)])

    Precision = []
    Recall = []
    for sigma in threshhold:
        predicted_positives = model_predictions >= sigma

        TP = np.sum((predicted_positives == True) & (true_classifications == True))
        FP = np.sum((predicted_positives == True) & (true_classifications == False))
        TN = np.sum((predicted_positives == False) & (true_classifications == False))
        FN = np.sum((predicted_positives == False) & (true_classifications == True))

        Precision.append(float(TP) / (TP + FP))
        Recall.append(float(TP) / (TP + FN))

    if make_plot:
        plt.plot(Recall, Precision)
        plt.xlabel('Recall', fontsize=15);
        plt.ylabel('Precision', fontsize=15);
        plt.title('PR-curve', fontsize=15);
        sns.despine();

    AUC = np.abs(np.trapz(y=Precision, x=Recall))
    return Precision, Recall, AUC

