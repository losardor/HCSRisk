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
        self.Capacity = self.NumOfPatients*(1+C)

    
    def not_disconnected(self, adj, failed, verbose=False):
        '''
        Checks whether the doctors are connected to any other throught the patient-sharing network.
        If not they are removed since they will not participate in the simulation (ignoring teleporation)
        '''
        lost=0
        hasOutDegree = np.any(adj, axis=0).transpose()
        hasInDegree = np.any(adj, axis=1)
        if verbose:
            print(hasInDegree.shape, hasOutDegree.shape)
        not_disconnected = np.asarray(np.logical_and(hasOutDegree,hasInDegree)).squeeze()
        self.originalID = self.originalID[not_disconnected.squeeze()]
        if verbose:
            print(self.originalID)
        disconnectedFailed = np.logical_not(np.in1d(failed, self.originalID))
        lost+=self.NumOfPatients[failed[disconnectedFailed]].sum()
        failed = failed[np.logical_not(disconnectedFailed)]
        if verbose:
            print('lost:',lost, 'Not disconnected out of the failed:', failed)
        self.NumOfPatients = self.NumOfPatients[not_disconnected.squeeze()]
        self.Capacity = self.Capacity[not_disconnected.squeeze()]
        return lost, failed

    def reindex(self, verbose = False, failed = []):
        '''
        Make a new index that starts at 0 and counts up to the active connected doctors.
        It returns a map from the old indices to the new (stored under IdForSimulation)
        '''
        self.IdForSimulation = np.arange(self.originalID.shape[0])
        if verbose:
            print("original failed: ", failed)
        doc2sim = {Id:newID for Id, newID in zip(self.originalID, self.IdForSimulation)}
        if verbose:
            print(doc2sim)
        failed = [doc2sim[f] for f in failed if f in doc2sim.keys()] + [(f,False) for f in failed if f not in doc2sim.keys()]
        if verbose:
            print(self.IdForSimulation, failed)
        return doc2sim