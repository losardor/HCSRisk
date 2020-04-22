#!/usr/bin/env python
# coding: utf-8


from Dynamics import *
import pandas as pd


def NNselfAvoiding(bez, C=0.1, predetermined = False):


    adj,mean_patient, docIds = LoadMatData(22)

    docs = doctors(len(docIds), mean_patient, C, ids = list(docIds))

    failed = remove_doctors(docs.originalID, 1, predetermined=predetermined)

    not_disconnected, lost, failed = docs.not_disconnected(adj, failed)

    adj = adj[not_disconnected, :]
    adj = adj[:,not_disconnected]

    doc2sim, failed = docs.reindex(verbose=False, failed=failed)

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