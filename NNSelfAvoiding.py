#!/usr/bin/env python
# coding: utf-8


from Dynamics import *
import pandas as pd


def NNselfAvoiding(bez, C=0.05, predetermined = False, verbose=False):
    

    adj,mean_patient, docIds = LoadMatData(bez)

    docs = doctors(len(docIds), mean_patient, C, ids = list(docIds))
    failed = remove_doctors(docs.originalID, 1, predetermined=predetermined)
    if verbose:
        print("Ids of doctors participating: ", docs.originalID)
    if verbose:
        print("ID of failed", failed)
    not_disconnected, lost, failed = docs.not_disconnected(adj, failed, verbose = False)
    if verbose:
        print("Doctors are connected? ", not_disconnected, "ID of failed if connected ", failed)
    if not failed:
        return {'locations':[], 'displacements':[], 'status':[]}, docs, lost
    adj = adj[not_disconnected, :]
    adj = adj[:,not_disconnected]

    doc2sim, failed = docs.reindex(verbose=False, failed=failed)
    if verbose:
        print("Reindexed failed: ", failed)
    patient = {}
    patient["locations"] = np.hstack(np.array([i*np.ones(NPats) 
                                        for i, NPats in zip(failed,
                                                            docs.NumOfPatients[failed])])).astype(int).squeeze()
    if verbose:
        print("There are ", len(patient["locations"]), " displaced patients")
    patient["status"] = np.ones(patient["locations"].shape, dtype=bool)
    patient["displacements"] = np.zeros(patient["locations"].shape)
    if verbose:
        plt.bar(np.arange(len(docs.NumOfPatients)),docs.NumOfPatients)
        plt.show()
    adj[:,failed] = 0
    t = 1
    while True:
        patient, docs, lost = step(patient, adj, docs, lost, verbose = verbose)
        t+=1
        if not np.any(patient["status"]):
            break

    return patient, docs, lost