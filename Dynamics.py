import pandas as pd
from scipy.io import loadmat
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def LoadMatData(path="./", docIds = np.arange(50), verbose = False):
    '''
    Loads the data collated during the previous work
    If the data is not in this folder it needs a path 
    to the storage folder. It also needs the indices (peterID)
    for the simulations
    '''

    Adj = loadmat(path+'Adj.mat')
    patient = loadmat('patientNums.mat')
    adj = Adj['Adj'][np.ix_(docIds, docIds)]
    if verbose:
        print(adj)
    adj = adj.todense()
    np.fill_diagonal(adj,0)
    mean_patient = patient['MP'][docIds]
    return adj, mean_patient


def remove_doctors(doctors, N, predetermined = False):
    '''
    Selects random doctors to be removed
    '''

    if predetermined:
        return doctors[predetermined]
    else:
        return np.random.choice(doctors, N)

class doctors:
    '''
    Class with properties and methods relating to the set of doctors
    '''

    def __init__(self, docNum, mean_patient, C, ids = False):

        if ids:
            self.originalID = ids
            self.NumOfPatients = mean_patient[ids].astype(int).squeeze()
        else:
            self.originalID = np.arange(docNum).squeeze()
            self.NumOfPatients = mean_patient[np.arange(docNum).squeeze()].astype(int).squeeze()
        self.capactiy = self.NumOfPatients*(1+C)

    
