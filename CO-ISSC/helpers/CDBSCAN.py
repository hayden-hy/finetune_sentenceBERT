from __future__ import annotations
from typing import Literal
from scipy.spatial import KDTree
import numpy as np
from scipy.spatial.distance import cdist
import math
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# from helpers.CDBSCAN import Point
from .Terms import CoreConcept, Term, Vocabulary

# The `Point` class represents a point in a clustering algorithm, with attributes such as its
# position, labels, and links to other points.
class Point():
    '''Point representing a term and its position in the embedding space.

    Contains a list of every other points that must be linked to it 
    and a list of every other points that cannot be linked to it.
    
    Parameters
    ----------
    term : Term
        Term associated to this point.
    index : int
        Term's index in the vocabulary.
    position : list
        Term's position in the embedding space.
    must_link : set
        Set that contains pairs of indices that must be in the same cluster.
        In other words, if two instances have indices that are in the `must_link` set, they should be
        assigned to the same cluster.
    cannot_link : set
        Set that contains pairs of instances that should not be assigned to the same cluster.
        It represents the constraints that certain instances should be kept separate
        from each other in the clustering process.
    
    '''
    def __init__(self, term : Term, index: int, position : list, must_link : set, cannot_link : set) -> None:
        self.term           : Term          = term
        self.index          : int           = index
        self.position       : list          = position
        self.labeled        : bool          = False

        self.must_link      : set           = must_link
        self.cannot_link    : set           = cannot_link

        self.cluster        : Cluster       = None
        self.core_cluster    : CoreCluster   = None

    
    def set_cluster(self, cluster : Cluster) :
        self.cluster = cluster
        self.labeled = True

    
    def __eq__(self, __value: object) -> bool:
        return self.index == __value.index


    def __hash__(self) -> int:
        return hash(self.index)


class Cluster():
    '''A cluster of Points
    
    Parameters
    ----------
    points : list[Point]
        List of Points contained in the cluster.
    
    '''
    def __init__(self, index: int, points : list[Point] = None) -> None:
        self.points         : set[Point]    = set()
        self.center         : np.ndarray    = None
        self.index          : int           = index

        self.must_link      : set(Point)    = set()
        self.cannot_link    : set(Point)    = set() 


        if points != None :
            self.add_all_points(points)
            self.__compute_center()


    def add_all_points(self, points: list[Point]):
        for point in points :
            self.add_point(point, False)

        self.__compute_center()


    def add_point(self, point : Point, recompute_center = True) :
        self.points.add(point)
        point.set_cluster(self)
        if recompute_center :
            self.__compute_center()
        
        self.must_link = self.must_link.union(point.must_link)
        self.cannot_link = self.cannot_link.union(point.cannot_link)

    
    def __compute_center(self):
        self.center = np.average([p.position for p in self.points], axis=0)


    def __eq__(self, __value: object) -> bool:
        if __value == None:
            return False
        return self.index == __value.index


    def __hash__(self) -> int:
        return hash(self.index)


class CoreCluster(Cluster):
    '''The above function is a constructor for a class that initializes various attributes including
        clusters, index, core concept, and center.
    
    Parameters
    ----------
    core_concept : CoreConcept
        The core concept associated with the cluster.
    center : list[float]
        The coordinates of the cluster's core concept.
    index : int
        Index of the core concept that will be used as the cluster value.
    points : set[Point]
        List of Points contained in the cluster.
    
    '''
    def __init__(self, core_concept : CoreConcept, center : list[float],  index: int, points: set[Point] = None) -> None:
        self.clusters_in_core_cluster = set()
        self.__get_local_clusters(points)
        super().__init__(index, points)
        # self.index : int = index
        self.core_concept = core_concept
        self.center = center

    
    def add_point(self, point: Point, recompute_center=False):
        point.core_cluster = self
        return super().add_point(point, recompute_center)


    def merge_cluster(self, cluster : Cluster):
        self.clusters_in_core_cluster.add(cluster)
        self.add_all_points(cluster.points)


    def __compute_center(self):
        return


    def __get_local_clusters(self, points):
        for point in points :
            self.clusters_in_core_cluster.add(point.cluster)
    

    def __eq__(self, __value: object) -> bool:
        if __value == None:
            return False
        return self.core_concept == __value.core_concept


    def __hash__(self) -> int:
        return hash(self.core_concept)


class CDBSCAN():
    '''Modified version of the original C-DBSCAN: 
    Density-Based Clustering with Constraints from Carlos Ruiz, Myra Spiliopoulou & Ernestina Menasalvas

    The modification only concern the must link part where we force the musk link relations to only be with
    our core concepts.
    
    Parameters
    ----------
    epsilon : float, optional
        Distance threshold that determines the maximum distance between two points for them to be considered neighbors.
        Points within this distance of each other are considered to be part of the same cluster.
        Increasing the value of epsilon will result in more points being included in each cluster.
    min_points : int, optional
        Minimum number of points required to form a cluster. 
        Any cluster with fewer points than this threshold will be considered noise.
    
    '''
    def __init__(self, epsilon : float = 1, min_points : int = 1) -> None:

        self.epsilon = epsilon
        self.min_points = min_points

        self.points                         = []    # List of points coordinates
        self.core_concepts_indices          = []    # List of each core concept index in the points array

        self.leaves                         = None  # List of the KDTree's leaves. Each leaf contains the coordinates of the points present in this leaf.
        self.leaves_idx                     = None  # List of the KDTree's leaves. Each leaf contains the index of the points present in this leaf.

        self.local_clusters                 = []    # List of the clusters formed at step 2 of the algorithm

        self.noise_cluster                  = None

        self.core_local_clusters            = []    # List of local clusters around a core concept


    def fit_predict(self, X, vocabulary : Vocabulary, core_concept_indices, last_iter = True):
        '''Compute the clustering and returns the predicted labels
        
        Parameters
        ----------
        X : array_like
            The input data, which is a matrix or array-like object of shape (n_samples, n_features). Each row
            represents a sample and each column represents a feature.
        vocabulary : Vocabulary
            Collection of terms and their relations that will be used in the clustering algorithm.
        core_concept_indices
            List of indices that represent the core concept's positions in the `vocabulary`.
            These indices indicate which terms in the vocabulary are considered as core concepts.
        last_iter : bool, optional
            Boolean flag that indicates whether the current iteration is the last iteration of the clustering algorithm.
            If set to True, the distance threshold will be ignored, to cluster every single point.
        
        Returns
        -------
            the transformed data.
        
        '''
        self.core_concepts_indices = core_concept_indices
        self.__setup_points(X, vocabulary)

        other_term_index = vocabulary.get_index("other")
        self.noise_cluster = CoreCluster(vocabulary.get_at(other_term_index), self.points[other_term_index].position, vocabulary.get_at(other_term_index).get_core_concept(), set([self.points[other_term_index]]))
        
        self.fit(X, vocabulary, last_iter)

        transformed_data = self.transform(last_iter,vocabulary.get_at(other_term_index).get_core_concept())

        X = np.array([point.position for point in self.points])
        self.__compute_metric(X, transformed_data)

        return transformed_data

    
    def fit(self, X, vocabulary : Vocabulary, last_iter : bool = True):
        '''The `fit` function fits a KDTree model to the input data, creates local clusters, and merges them
        based on must-link and cannot-link constraints.
        
        It follows the steps 1 to 3b from the C-DBSCAN: Density-Based Clustering with Constraints paper.

        Parameters
        ----------
        X : array_like
            The input data, which is a matrix or array-like object of shape (n_samples, n_features). Each row
            represents a sample and each column represents a feature.
        vocabulary : Vocabulary
            Collection of terms and their relations that will be used in the clustering algorithm.
        core_concept_indices
            List of indices that represent the core concept's positions in the `vocabulary`.
            These indices indicate which terms in the vocabulary are considered as core concepts.
        last_iter : bool, optional
            Boolean flag that indicates whether the current iteration is the last iteration of the clustering algorithm.
            If set to True, the distance threshold will be ignored, to cluster every single point.
        
        '''
        # Step 1
        self.kd_tree = KDTree(X, leafsize=self.min_points * 3)

        # Intermediate steps to prepare the data for step 2
        self.leaves_idx = self.__get_leaves(self.kd_tree.tree)

        self.leaves = []
        self.__remap_leaves_to_data()

        # Step 2
        self.__create_local_cluster()
        # Step 3a
        self.__merge_local_clusters(vocabulary.must_link)
        # Step 3b
        self.__merge_clusters(last_iter)


    def transform(self,last_iter : bool = False,other_cc_index : int = None):
        y = np.full(len(self.points), -1)
        for i, p in enumerate(self.points):
            # print(p.term)
            if p.core_cluster != None :
                y[i] = p.core_cluster.index

            # if last iteration, assign the unclustered points to "other"
            elif last_iter: 
                y[i] = other_cc_index
        return y

    
    def __create_local_cluster(self, last_iter : bool = True):
        '''Step 2 : Create local clusters based on a set of leaf points, considering cannot-link
        constraints and minimum point requirements.
        
        Checks every leaf of the KDTree

        For evey leach, checks if a cannot-link relation is present between at least two terms
        from the leaf. If so, create a cluster for each point

        If no cannot-link relations exist, checks for every point its neighbors.
        Either assign the point to the noise cluster or create a new cluster, based
        on the number of neighbors.

        Parameters
        ----------
        last_iter : bool, optional
            Boolean flag that indicates whether the current iteration is the last iteration of the clustering algorithm.
            If set to True, the distance threshold will be ignored, to cluster every single point.
        
        '''
        for leaf_points_idx in self.leaves_idx :
            leaf_cannot_link = set()
            for point_idx in leaf_points_idx :
                leaf_cannot_link = leaf_cannot_link.union(self.points[point_idx].cannot_link)
            
            if any(point_idx in leaf_cannot_link for point_idx in leaf_points_idx) :
                for point_idx in leaf_points_idx :
                    points = [self.points[point_idx]]
                    self.__add_new_cluster(points)
                continue

            for point_idx in leaf_points_idx :
                point = self.points[point_idx]
                if point.labeled:
                    continue
                
                indices = [idx for idx in leaf_points_idx
                           if math.dist(point.position, self.points[idx].position) <= self.epsilon]

                if len(indices) < self.min_points :
                    self.noise_cluster.add_point(point)
                    point.labeled = True
                    
                else :
                    points = [self.points[i] for i in indices]
                    self.__add_new_cluster(points)


    def __merge_local_clusters(self, must_link : dict[int, list[int]]):
        '''Step 3a : Merges local clusters based on must-link constraints, forming the core clusters.

        As the must link are always defined around a core concept, we create a core cluster for
        each core concept and then assign every local cluster to it, based on the must link relations.
        
        Parameters
        ----------
        must_link : dict[int, list[int]]
            Dictionary mapping core concept indices to terms indices.
        
        '''
        for cc_index_in_terms in self.core_concepts_indices:
            core_concept = self.points[cc_index_in_terms]
            index = core_concept.term.get_core_concept()
            points = [self.points[cc_index_in_terms]]
            if cc_index_in_terms in must_link :
                points += [self.points[index] for index in must_link[cc_index_in_terms]]
            core_cluster = CoreCluster(core_concept.term, core_concept.position, index, points)
            self.core_local_clusters.append(core_cluster)
    

    def __merge_clusters(self, last_iter : bool = False):
        '''Merges clusters in a clustering algorithm, specifically by finding
        the closest local cluster to each core cluster and merging them together.
        
        Parameters
        ----------
        last_iter : bool, optional
            Boolean flag that indicates whether the current iteration is the last iteration of the clustering algorithm.
            If set to True, the distance threshold will be ignored, to cluster every single point.
        
        '''
        previous_NLC = 1
        new_NLC = 0

        while previous_NLC != new_NLC :
            cluster_to_merge = list(set(point.cluster for point in self.points if point.core_cluster == None))

            previous_NLC = len(cluster_to_merge)
            new_NLC = previous_NLC

            for core_cluster in self.core_local_clusters :
                cluster_to_merge = list(set(point.cluster for point in self.points if point.core_cluster == None))

                if (len(cluster_to_merge) == 0) :
                    continue

                closest_local_cluster = self.__get_closest_local_cluster_index(core_cluster, cluster_to_merge, last_iter)

                if closest_local_cluster == None:
                    continue
                
                core_cluster.merge_cluster(closest_local_cluster)

                new_NLC = len(cluster_to_merge) - 1


    def __get_closest_local_cluster_index(self, core_cluster : CoreCluster, possible_clusters : list[Cluster], last_iter : bool = False):
        '''Finds the closest local cluster to a given core cluster based on their centers and a distance threshold.
        
        Parameters
        ----------
        core_cluster
            Core cluster to which the nearest cluster will be merged.
        possible_clusters
            A list of non yet merged clusters.
        last_iter : bool, optional
            Boolean flag that indicates whether the current iteration is the last iteration of the clustering algorithm.
            If set to True, the distance threshold will be ignored, to cluster every single point.
        
        Returns
        -------
            the closest local cluster to the given core cluster, based on certain conditions.
        
        '''
        LIMIT = np.Inf if last_iter else 0.75

        core_cluster_center = [core_cluster.center]
        possible_clusters_centers = [cluster.center for cluster in possible_clusters]

        distances = cdist(core_cluster_center, possible_clusters_centers)[0]

        valid_idx = np.where(distances <= LIMIT)[0]
        if len(valid_idx) == 0 :
            return None

        sorted_index = np.argsort(distances)

        core_cluster_cannot_link = [self.points[index] for index in core_cluster.cannot_link]

        closest_local_cluster = None
        for i in range(len(possible_clusters)):
            closest_local_cluster = possible_clusters[sorted_index[i]]

            if any(p in core_cluster_cannot_link for p in closest_local_cluster.points):
                continue

            return closest_local_cluster
        return None

    def __setup_points(self, X : list[list[float]], vocabulary : Vocabulary):
        """Map every term to a Point instance

        Args:
            X (list[list[float]]): The position of every term in the embedding
            vocabulary (Vocabulary): The vocabulary instance to map a term to its position
        """
        for term_index in vocabulary.terms_idx.values() :
            term_must_link = set()
            if term_index in vocabulary.must_link :
                term_must_link = vocabulary.must_link[term_index]
            
            term_cannot_link = set()
            if term_index in vocabulary.cannot_link :
                term_cannot_link = vocabulary.cannot_link[term_index]

            index = len(self.points)
            point = Point(vocabulary.terms[term_index], index, X[term_index], term_must_link, term_cannot_link)
            self.points.append(point)


    def __compute_metric(self, X, y, metric : Literal["silhouette"] | Literal["calinski-harabasz"] | Literal["davies-bouldin"] = "silhouette"):
        if metric == "silhouette" :
            s = silhouette_score(X, y, metric="euclidean")
            print(s)
            return
        
        if metric == "calinski-harabasz" :
            s = calinski_harabasz_score(X, y)
            print(s)
            return
        
        if metric == "davies-bouldin" :
            s = davies_bouldin_score(X, y)
            print(s)
            return


    def __add_new_cluster(self, points : list[Point]) -> CDBSCAN:
        """Utilitary method to create a new cluster

        Args:
            points (list[Point]): List of points in the cluster

        Returns:
            CDBSCAN : Current instance of CDBSCAN
        """
        index = len(self.local_clusters)
        cluster = Cluster(index, points)
        self.local_clusters.append(cluster)
        return self


    def __get_leaves(self, node : KDTree.node) -> list[list[int]]:
        """Utilitary method to recursively get every leaf of the KDTree

        Args:
            node (KDTree.node): Current node to be treated

        Returns:
            list[int]: List of every points contained in every leaf.
        """
        if not hasattr(node, "greater") and not hasattr(node, "less") :
            return [node.idx]
        
        leaves = []
        
        if hasattr(node, "greater") :
            leaves += self.__get_leaves(node.greater)

        if hasattr(node, "less") :
            leaves += self.__get_leaves(node.less)
        
        return leaves
    

    def __remap_leaves_to_data(self) -> CDBSCAN:
        self.leaves = []
        for leaf_idx in self.leaves_idx :
            self.leaves.append(list(map(lambda idx: self.points[idx], leaf_idx)))
        
        return self