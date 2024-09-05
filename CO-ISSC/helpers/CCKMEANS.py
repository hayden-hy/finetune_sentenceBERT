from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from .Terms import Vocabulary

class CCKMEANS():

    # def __init__(self,n_) -> None:

    def fit_predict(self, X, vocabulary : Vocabulary, core_concept_indices, last_iter = True):
        # Predict the cluster index for each term surrounding the core concepts.

        other_index = vocabulary.get_index("other")
        center_indices = np.append(core_concept_indices,other_index) #core_concept_indices
        for i in center_indices:
            print(vocabulary.get_at(i))
            print(vocabulary.get_at(i).get_core_concept())
        self.kmeans = KMeans(n_clusters=7, init=X[center_indices],random_state=0)
        # print([vocabulary.get_at(index) for index in core_concept_indices])

        # Fit the KMeans model
        pseudo_preds = self.kmeans.fit_predict(X)
        print(self.kmeans.n_iter_)

        # assign the lable to the core conceptt label
        preds = np.zeros(len(X),dtype=int)
        for i,pred in enumerate(pseudo_preds):
            preds[i] = vocabulary.get_at(center_indices[pred]).get_core_concept()


        return preds
