#!/usr/bin/env python
# coding: utf-8


from Dynamics import *
import pandas as pd


def NNselfAvoiding(bez, C=0.1, predetermined = False, verbose=False):
    

    adj,mean_patient, docIds = LoadMatData(bez)

    docs = doctors(len(docIds), mean_patient, C, ids = list(docIds))
    failed = remove_doctors(docs.originalID, 1, predetermined=predetermined)
    if verbose:
        print(docs.originalID)
    if verbose:
        print(failed)
    not_disconnected, lost, failed = docs.not_disconnected(adj, failed, verbose = False)
    if verbose:
        print(not_disconnected, failed)
    if not failed:
        return {'locations':[], 'displacements':[], 'status':[]}, docs, lost
    adj = adj[not_disconnected, :]
    adj = adj[:,not_disconnected]

    doc2sim, failed = docs.reindex(verbose=False, failed=failed)
    if verbose:
        print(failed)
    patient = {}
    patient["locations"] = np.hstack(np.array([i*np.ones(NPats) 
                                        for i, NPats in zip(failed,
                                                            docs.NumOfPatients[failed])])).astype(int).squeeze()
    patient["status"] = np.ones(patient["locations"].shape, dtype=bool)
    patient["displacements"] = np.zeros(patient["locations"].shape)

    adj[:,failed] = 0
    t = 1
    while True:
        patient, docs, lost = step(patient, adj, docs, lost, verbose = False)
        t+=1
        if not np.any(patient["status"]):
            break

    return patient, docs, lost