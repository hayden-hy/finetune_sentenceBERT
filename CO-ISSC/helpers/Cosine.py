from __future__ import annotations
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Point:
    def __init__(self, term_idx, embedding):
        self.term_idx = term_idx # the term index
        self.embedding = embedding # the embedding of the point
        self.cc_idx = None # the core concept index of the point
    
    
class CoreCluster:
    def __init__(self, cc_idx, point)-> None:
        self.cc_idx = cc_idx # the core concept index
        self.cluster = [] 
        self.cluster_center = point.embedding # the embedding of the core concept
        self.add_point(point)

    def add_point(self, point):
        self.cluster.append(point)
        point.cc_idx = self.cc_idx

class Cosine:
    """ Similarity Measure Based Method :
    Clustering the terms according to the similarity measure between the term embeddings and the core concept embeddings.
    """
    
    def __init__(self):
        pass
    def fit_predict(self, X, vocabulary,cc_indices,last_iter):
        self.predict(X,vocabulary,cc_indices,last_iter)
        pred = self.transform()
        print(pred)
        return pred

    def predict(self,X,vocabulary,cc_indices,last_iter):
        """
        Predict the cluster index for each term surrounding the core concepts.
        """
        self.points = self.__init_points(X)
        self.coreclusters = self.__init_coreclusters(self.points,vocabulary,cc_indices)

        cosines = cosine_similarity(X,[corecluster.cluster_center for corecluster in self.coreclusters])

        for i,point in enumerate(self.points):
            if point.cc_idx is None:
                idx = np.argsort(cosines[i])[-1]
                self.coreclusters[idx].add_point(point)


    def transform(self):
    
        return np.array([point.cc_idx for point in self.points])
        
    def __init_points(self,X):
        """
        Initialize the points with the term indices and the embeddings.
        """
        points = []
        for i in range(len(X)):
            point = Point(i,X[i])
            points.append(point)
        return points
    
    def __init_coreclusters(self,points,vocabulary,cc_indices):
        """
        Initialize the core clusters with the core concept indices and the embeddings.
        """
        coreclusters = []
        print(vocabulary.core_concepts)
        for i in range(len(vocabulary.core_concepts)):
            cc_idx = vocabulary.get_index(vocabulary.core_concepts[i]) 
            corecluster = CoreCluster(i,points[cc_idx])
            coreclusters.append(corecluster)

        return coreclusters
    
