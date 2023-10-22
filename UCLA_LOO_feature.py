from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split
from scipy.stats import pearsonr
import numpy as np
import time
from sklearn import linear_model
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFpr
import matplotlib.pyplot as plt

NB_ROI = 100
NB_SUBJECTS = 99

def prediction_lassocv(features_initial, ability, b, N_sub):
    """
    Ability prediction model.
    Input: threshold: 'weak' or 'strong' or None, will keep only the specified subset of links (weak, strong or full)
                      In the paper, I don't use only strong and None for plot 3
           b: list containing all values of the beta exponent (the lasso regression will be done for each independently)
           iter: the number of iteration (since the method is stochastic, we want to avoid any lucky iteration so we average over many)

    Output: save files containing the weights of retained links as well as the correlation between predicted and empirical for each iteration and each value of beta
    """
    all_corr = []
    reg = LassoCV(cv=5, n_jobs=1)
    reg=LinearRegression()
    for beta in b:  # Parameter beta loop
        predicted = np.zeros(np.size(ability))
        print("beta " + str(beta))
        # Use the same training and testing folds for different values of beta
        features = np.power(features_initial, beta)
        SC_global=np.mean(features,axis=1)
        features=np.append(features,SC_global.reshape(N_sub,1),axis=1)
        k_fold = KFold(np.size(ability))
        for k, (train, test) in enumerate(k_fold.split(features, ability)):  # Inner k-fold loop
            correlations = np.zeros(np.size(features[0]))
            p_values = np.zeros(np.size(features[0]))
            for i in range(0, np.size(features[0])):
                weights = features[train, i]
                scores = ability[train]
                if len(weights[weights != 0]) < 3:  # in case of rare instances where no common significantly correlated links between LOOs are found
                    correlations[i] = 0
                    p_values[i] = 0
                else:
                    correlations[i], p_values[i] = pearsonr(weights[weights != 0], scores[
                           weights != 0])  # Some weights may be 0, therefore constant and return a warning
            to_remove = (p_values > 0.05).nonzero()  # Unsignificantly correlated links to remove
            features = np.delete(features, to_remove, axis=1)                          
        #xnew = SelectFpr(f_regression, alpha=0.01).fit_transform(features, ability)
        scores = cross_val_predict(reg, features, ability, cv=N_sub)
        cor = np.corrcoef(ability, scores)
        all_corr.append(cor[0,1])
    return all_corr

def main():
    """
    Main function used to call the other functions
    """
    ## Abilities:g	cry	mem	spd	PMAT24_A_CRz	VSPLOT_TCz	PicSeq_Unadjz	IWRD_TOTz	PicVocab_Unadjz	ReadEng_Unadjz	CardSort_Unadjz	Flanker_Unadjz	ProcSpeed_Unadjz
    UCLAabilities = np.genfromtxt('E:/Weak links project/Data/UCLAabilities.csv', delimiter=',')
    for k in range(0,np.size(UCLAabilities[1,:])):
        features_initial = np.genfromtxt('E:/Weak links project/Data/prediction/processed/UCLA_SC.csv', delimiter=',')
        ability=UCLAabilities[:,k]
        N_sub=99
        nan_index=np.where(np.isnan(ability))
        ability= np.delete(ability, nan_index, axis=0)
        features_initial= np.delete(features_initial, nan_index, axis=0)
        
        N_sub=N_sub-np.shape(nan_index)[1]

        b=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
        all_corr= prediction_lassocv(features_initial, ability, b, N_sub)
        np.savetxt('E:/Weak links project/Data/computed/UCLA_ability_'+str(k+1)+'_corr_LOO.csv', np.asarray(all_corr),
                   delimiter=',')
    plt.plot(all_corr)
    plt.show()
    


if __name__== "__main__":
    main()
