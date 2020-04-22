import pandas as pd
from scipy.io import loadmat
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def LoadMatData(bezID, path="./", verbose = False):
    '''
    Loads the data collated during the previous work
    If the data is not in this folder it needs a path 
    to the storage folder. It also needs the indices (peterID)
    for the simulations
    '''

    vBez = pd.read_csv(path+"vBez.csv", names = ['bez'])
    vFG = pd.read_csv(path+"vFG.csv", names = ['kind'])

    docfilter = vFG['kind'].isin([1,7,8]) #Filter out non primary care providers
    mapfilter = vBez['bez'] == bezID
    docIds = np.arange(len(docfilter))
    docIds = docIds[np.logical_and(docfilter.values.flatten(),mapfilter.values.flatten())]
    distnum = vBez[docfilter]
    Adj = loadmat(path+'Adj.mat')
    patient = loadmat('patientNums.mat')
    adj = Adj['Adj'][np.ix_(docIds, docIds)]
    if verbose:
        print(adj)
    adj = adj.todense()
    np.fill_diagonal(adj,0)

    mean_patient = patient['MP'][distnum == bezID]

    return adj, mean_patient, docIds


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
            self.originalID = np.array(ids)
            self.NumOfPatients = mean_patient.astype(int).squeeze()
        else:
            self.originalID = np.arange(docNum).squeeze()
            self.NumOfPatients = mean_patient.astype(int).squeeze()
        self.Capacity = self.NumOfPatients*(1+C)
        self.incoming = []
        self.availability = []
        self.Excess = []

    
    def not_disconnected(self, adj, failed, verbose=False):
        '''
        Checks whether the doctors are connected to any other throught the patient-sharing network.
        If not they are removed since they will not participate in the simulation (ignoring teleporation)
        '''
        lost=0
        hasOutDegree = np.any(adj, axis=0).transpose()
        hasInDegree = np.any(adj, axis=1)
        not_disconnected = np.asarray(np.logical_and(hasOutDegree,hasInDegree)).squeeze()
        newFailed = np.array(np.in1d(self.originalID,failed) ).nonzero()[0]
        disconnectedFailed = newFailed[np.logical_not(np.in1d(self.originalID[newFailed],self.originalID[not_disconnected]))]
        if verbose:
            print(hasInDegree.shape, hasOutDegree.shape, not_disconnected, disconnectedFailed)
        lost+=self.NumOfPatients[disconnectedFailed].sum()
        self.originalID = self.originalID[not_disconnected.squeeze()]
        if verbose:
            print(self.originalID, disconnectedFailed, self.NumOfPatients)
        failed = failed[np.logical_not(np.in1d(newFailed, disconnectedFailed))]
        if verbose:
            print('lost:',lost, 'Not disconnected out of the failed:', failed)
        self.NumOfPatients = self.NumOfPatients[not_disconnected.squeeze()]
        self.Capacity = self.Capacity[not_disconnected.squeeze()]
        return not_disconnected, lost, failed

    def reindex(self, verbose = False, failed = []):
        '''
        Make a new index that starts at 0 and counts up to the active connected doctors.
        It returns a map from the old indices to the new (stored under IdForSimulation)
        '''
        self.IdForSimulation = np.arange(self.originalID.shape[0])
        if verbose:
            print('original ID: ', self.originalID)
            print("original failed: ", failed)
        doc2sim = {Id:newID for Id, newID in zip(self.originalID, self.IdForSimulation)}
        if verbose:
            print(doc2sim)
        failed = [doc2sim[f] for f in failed if f in doc2sim.keys()] + [(f,False) for f in failed if f not in doc2sim.keys()]
        if verbose:
            print(self.IdForSimulation, failed)
        return doc2sim, failed


def vectorized(prob_matrix, items, oldpositions, verbose=False):
    ''''
    Given a probability matrix and a list of items it selects on item based on the column of probability matrix
    '''

    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    if verbose:
        print("k-shape: ",k.shape, "prob_matrik-shape: ",prob_matrix.shape, 
            "oldpositions-shape: ",oldpositions.shape, 
            "no_outlinks-IDs: ",np.nonzero(k == prob_matrix.shape[0]))
    try:
        k[k == prob_matrix.shape[0]] = oldpositions[np.asarray(k == prob_matrix.shape[0]).squeeze()] #necessary in the case there is no outlink
    except:
        print(k,prob_matrix.shape[0],oldpositions)
        sys.exit(1)
    return items[k]


def step(patient, adj, docs, lost, alpha = 0.15, maxSteps = 11, verbose = False):

    prob_weights = adj[patient["locations"][ patient["status"].astype(bool)] ] #Create Transport matrix
    #Assign target nodes to patient
    if prob_weights.shape[0] == 1:
        patient["status"] = np.zeros(patient["locations"].shape).astype(bool) #the single patient creates indexing problems while note being relevant
        return patient, docs, lost
    targets = vectorized(prob_weights.transpose(), docs.IdForSimulation, 
                         patient["locations"][ patient["status"].astype(bool)] , verbose)
    teleport = np.random.random(targets.shape)<alpha #Choose whether to teleport
    targets[teleport] = np.random.choice(docs.IdForSimulation) #Teleport patient
    patient["locations"][ patient["status"].astype(bool)] = targets.squeeze() #Update locations
    #Count docs.incoming"] patient
    # docs.incoming, _ = np.histogram(patient["locations"][ patient["status"].astype(bool)], 
    #     bins=np.arange(len(docs.IdForSimulation)+1))
    docs.incoming = np.zeros(docs.IdForSimulation.shape)
    at_doc = {}
    for doc in docs.IdForSimulation:
        #Determine indices of patient at location
        at_doc[doc] = np.asarray(patient["locations"][ patient["status"].astype(bool)] == doc).nonzero()[0]
        docs.incoming[doc] = len(at_doc[doc])
    docs.availability = (docs.Capacity-docs.NumOfPatients).squeeze().astype(int) #Calculate current availability
    docs.availability[docs.availability<0] = 0
    if verbose:
        sns.distplot(docs.availability)
        plt.show()
    #Compute how many patient will need to continue their search
    docs.Excess = docs.incoming-docs.availability 
    Absorbed = (docs.Excess<=0) #Check whether doctors have absorbed their load
    #Doctors that have absorbed their load need not send out patient (these would be negative numbers)
    docs.Excess[Absorbed] = 0 
    #Compute number of patient per doctor
    docs.NumOfPatients[Absorbed] =  docs.NumOfPatients[Absorbed] + docs.incoming[Absorbed]
    #Fill up doctors that have not absorbed the load
    docs.NumOfPatients[np.logical_not(Absorbed)] = docs.Capacity[np.logical_not(Absorbed)]
    #Patients at doctor that have absorbed can stop
    
    patient["displacements"][ patient["status"].astype(bool) ]+=1 #Update number of displacements
    # Randomply pick patient that can stay at doctors that have filled up
    for doc in docs.IdForSimulation: 
        # Skip doctors with now patient docs.incoming"]
        if docs.incoming[doc] and not Absorbed[doc] and docs.availability[doc] != 0:
            #Randomly pick the patient
            try:
                kept = np.random.choice(at_doc[doc], size = docs.availability[doc], replace = False)
            except:
                print('#at_doc: ',len(at_doc[doc]), 'doc availability: ', docs.availability[doc], 
                '#incoming',docs.incoming[doc], 'doc: ',doc)
            if verbose:
                print("at_doc-shape: ", at_doc.shape, "availability: ", docs.availability[doc].astype(int))
                print(kept.size)
            patient["status"][kept] = 0 #Change status of the lucky ones
    patient["status"][ np.in1d(patient["locations"], docs.IdForSimulation[Absorbed]) ] = 0 
    patient["status"][ patient["displacements"]>=maxSteps ] = 0
    lost+= np.array(patient["displacements"]>=maxSteps).sum()
    if verbose:
        print("Active Patients",patient["status"].sum())
    return patient, docs, lost