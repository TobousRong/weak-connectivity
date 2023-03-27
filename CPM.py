
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
import numpy as np
import random
import time
import sys
import operator
import os


NB_ROI = 360
NB_SUBJECTS = 991

def prediction_lassocv(b=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2], iter=100):
    """
    Ability prediction model.
    Input: threshold: 'weak' or 'strong' or None, will keep only the specified subset of links (weak, strong or full)
                      In the paper, I don't use only strong and None for plot 3
           b: list containing all values of the beta exponent (the lasso regression will be done for each independently)
           iter: the number of iteration (since the method is stochastic, we want to avoid any lucky iteration so we average over many)

    Output: save files containing the weights of retained links as well as the correlation between predicted and empirical for each iteration and each value of beta
    """
    #[ability].csv contain 1d vector of lower triangle for each subject (since there are some missing data in HCP, there are different number of subjects in every csv)
    features_initial = np.genfromtxt('E:/Weak links project/Data/prediction/processed/SC.csv', delimiter=',')
    ability = np.genfromtxt('E:/Weak links project/Data/HCP_abilities/g.csv', delimiter=',')
    
    avg = features_initial.sum(axis=0)/np.size(ability) #Compute average weight to threshold
    
    all_corr = []
    all_p = []
    retained = [[] for Null in range(12)]
    retained_weight = [[] for Null in range(12)]
    id_retained = [[] for Null in range(12)]
    print(len(retained))
    for ite in range(iter): # Iteration loop
        #random.seed(ite)
        #np.random.shuffle(ability)        
        start = time.time()
        print("iter "+str(ite))
        #Split into training and testing, KFold always return random folds
        k_fold_outer = KFold(2, shuffle=True)
        iter_corr = [] #List containing all final correlations, one for every beta
        iter_p = []
        for beta in b: # Parameter beta loop
            predicted = np.zeros(np.size(ability))
            print("beta "+str(beta))
            #Use the same training and testing folds for different values of beta
            features = np.power(features_initial, beta)
            #Defining the lasso model
            lasso = LassoCV(cv=5, max_iter=5000, normalize=False, tol=10e-4, n_jobs=-1) #, random_state=0
            #Perform cpm feature extraction using leave one out on training set
            for train_outer,test_outer in k_fold_outer.split(features,ability): #Outer k-fold loop
                idc = np.tril_indices(NB_ROI,k=-1)
                idc = list(zip(idc[0],idc[1])) #lower tril list of tuples (x,y) to trace retained links
                idc = np.asarray(idc)
                
                prediction_fail = False #In case feature selection fail. we need to handle it
                features_train = features[train_outer]
                ability_train = ability[train_outer]
                features_test = features[test_outer]
                ability_test = ability[test_outer]
                average = avg # Variable to keep track of average weights for retained links
                k_fold = KFold(np.size(ability_train))
                for k, (train,test) in enumerate(k_fold.split(features_train,ability_train)): #Inner k-fold loop
                    correlations = np.zeros(np.size(features_train[0]))
                    p_values = np.zeros(np.size(features_train[0]))
                    for i in range(0,np.size(features_train[0])):
                        weights = features_train[train,i]
                        scores = ability_train[train]
                        if len(weights[weights!=0])<3: #in case of rare instances where no common significantly correlated links between LOOs are found
                            correlations[i] = 0
                            p_values[i] = 0
                        else:
                            correlations[i], p_values[i] = pearsonr(weights[weights!=0],scores[weights!=0]) #Some weights may be 0, therefore constant and return a warning
                    to_remove = (p_values>0.001).nonzero() #Unsignificantly correlated links to remove
                    features_train = np.delete(features_train, to_remove, axis=1)
                    features_test = np.delete(features_test, to_remove, axis=1)
                    average = np.delete(average, to_remove)
                    idc = np.delete(idc, to_remove, axis=0)
                    
                if np.shape(features_train)[1] != 0 :
                    lasso.fit(features_train, ability_train) #Train the Lasso regression model
                    predicted_ability = lasso.predict(features_test) #Test the model
                    predicted[test_outer] = predicted_ability
                    # print(pearsonr(predicted_ability, ability_test))
                    coeff = lasso.coef_
                    retained_avg_weights = average[coeff!=0]
                    n=np.nonzero(coeff)
                    retained_indices = idc[n]
                    coeff = coeff[n]
                    retained[int(beta*10-1)].append(retained_avg_weights)
                    retained_weight[int(beta*10-1)].append(coeff)
                    id_retained[int(beta*10-1)].append(retained_indices)
                else: #If there are no elements in feature train, it means no features survived the feature selection and prediction is not possible
                    prediction_fail = 1
            if prediction_fail:
                print("Prediction failed: no feature survived selection.")
                iter_corr.append(np.nan)
                iter_p.append(np.nan)
            else:
                corr, p = pearsonr(predicted, ability)
                iter_corr.append(corr)
                iter_p.append(p)
        all_corr.append(iter_corr)
        print(iter_corr)
        all_p.append(iter_p)
        end = time.time()
        print("Iteration time: " + str(end-start))
    for i,l in enumerate(retained): #Save the mean weight of retained links
        single_list = [weight for sublist in l for weight in sublist]
        np.savetxt('E:/Weak links project/Data/computed/full_beta_0'+str(i+1)+'.csv', np.asarray(single_list), delimiter=',')
    for i,l in enumerate(id_retained): #Save the id of retained links
        single_list = [weight for sublist in l for weight in sublist]
        np.savetxt('E:/Weak links project/Data/computed/full_beta_0'+str(i+1)+'_id.csv', np.asarray(single_list), delimiter=',')
    for i,l in enumerate(retained_weight): #Save the mean weight of retained links
        single_list = [weight for sublist in l for weight in sublist]
        np.savetxt('E:/Weak links project/Data/computed/full_beta_0'+str(i+1)+'_weight.csv', np.asarray(single_list), delimiter=',')
    np.savetxt('E:/Weak links project/Data/computed/corr_100iter_2f.csv', np.asarray(all_corr), delimiter=',')
    np.savetxt('E:/Weak links project/Data/computed/pval_100iter_2f.csv', np.asarray(all_p), delimiter=',')

def main():
    """
    Main function used to call the other functions
    """
    #prepare_data_lasso()
    prediction_lassocv()


if __name__== "__main__":
    main()
