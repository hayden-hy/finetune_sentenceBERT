from __future__ import annotations
from . import Term
from . import CoreConcept

import numpy as np

class Vocabulary():
    def __init__(self):
        """Utilitary class allowing to convert index to term and term to index

        terms : list of every terms contained in the vocabulary
        term_idx : dictionary mapping a term to his index in the variable "terms"
        must_link : graph linking an hypernym to its hyponyms that must link to it
        cannot_link : graph linking each core concept and their hyponyms to the other core concepts and their hyponyms
        core_concept : list containing every core concept used in 
        
        """
        self.terms          : list              = []
        self.terms_idx      : dict[str, int]    = {}
        self.must_link      : dict[int, set]    = {}
        self.cannot_link    : dict[int, set]    = {}
        self.core_concepts  : list              = []

        self.__index         : int               = -1 # Counter used in for the generator
    

    def __iter__(self):
        return self
    

    def __next__(self):
        """Built in function allowing to use the class as a generator
        
        Exemple :
        vocab = Vocabulary()

        for term in vocab :
            print(term)

        """
        self.__index += 1
        if self.__index >= len(self.terms):
            self.__index = -1
            raise StopIteration()
        return self.terms[self.__index]
    

    def append(self, term):
        """Append a term to the list

        If the term is already present it won't make a duplicate
        
        """
        if term not in self.terms :
            self.terms.append(term)
            self.terms_idx[term] = len(self.terms) - 1
    

    def get_term_from_string(self, string : str) -> Term:
        """Return the term instance corresponding to a string

        Args:
            string (str): string version of a term

        Raises:
            Exception: Term not present in the vocabulary

        Returns:
            Term: Term instance corresponding to a string
        """
        if string not in self.terms_idx:
            raise Exception("Term not present in the vocabulary")

        index = self.terms_idx[string]
        return self.terms[index]
    

    def get_index(self, term : str) -> int:
        """Get the index of a term in the vocabulary

        Args:
            term (str): Term or string version of a term

        Returns:
            int: Index of the term in the vocabulary
        """
        return self.terms_idx[term]
    

    def get_at(self, index : int) -> Term:
        """Return a term based on its index.
        Prefere using the array notation for clarity

        Args:
            index (int): Index of the term

        Returns:
            Term: Instance of the term
        """
        return self.terms[index]


    def add_must_link(self, hypernym : Term, hyponym : Term) -> Vocabulary:
        """Add a one way "must link" relation between two terms.
        The hypernym should be a core concept for better results

        Args:
            hypernym (Term): Term that will be linked to many other
            hyponym (Term): Term that will be merged 

        Returns:
            Vocabulary: Vocabulary instance
        """
        hypernym = self.get_index(hypernym)
        hyponym = self.get_index(hyponym)
        self.__add_element_to_dict(self.must_link, hypernym, hyponym)
        return self


    def add_cannot_link(self, term_a : Term, term_b : Term) -> Vocabulary:
        """Add a two way "cannot link" relation between two terms.
        The only relations created should be between core concepts or hyponyms of core concepts.

        Args:
            term_a (Term): First term of the relation
            term_b (Term): Second term of the relation

        Returns:
            Vocabulary: Vocabulary instance
        """
        self.__add_relation(self.cannot_link, term_a, term_b)
        return self


    def __add_relation(self, _dict : dict[str, set(Term)], term_a : Term, term_b : Term) -> Vocabulary:
        """Utility method to add a relation to a dictionary

        Args:
            _dict (dict[str, set): Dictionary containing the relations
            term_a (Term): First term of the relation
            term_b (Term): Second term of the relation

        Returns:
            Vocabulary: Vocabulary instance
        """

        index_a = self.get_index(term_a)
        index_b = self.get_index(term_b)

        self.__add_element_to_dict(_dict, index_a, index_b)
        self.__add_element_to_dict(_dict, index_b, index_a)
        return self

    
    def __add_element_to_dict(self, _dict : dict[str, set(Term)], key : Term, value : Term) -> Vocabulary:
        """Utilitay method to add a value to a dictionary
        The method create the set stored at each key of the dictionary if not present

        Args:
            _dict (dict[str, set): Dictionary containing the relations
            key (Term): Key of the dictionary
            value (Term): The value to add to the set contained at the key

        Returns:
            Vocabulary: Vocabulary instance
        """
        if key not in _dict:
            _dict[key] = set()
        
        _dict[key].add(value)
        return self
    

    def update_vocabulary(self, y_pred : np.ndarray) -> Vocabulary:
        """Update the vocabulary based on the result of a model's prediction

        Args:
            y_pred (np.ndarray): Result of a prediction 

        Returns:
            Vocabulary: Vocabulary instance
        """
        self.update_cannot_link(y_pred)
        self.update_must_link(y_pred)


    def update_must_link(self, y_pred : np.ndarray) -> Vocabulary:
        """Modify the must link relations based on the prediction
        Associate every word to its predicted core concept

        Args:
            y_pred (np.ndarray): Result of a prediction 

        Returns:
            Vocabulary: Vocabulary instance
        """
        self.must_link = {}
        for term_index, cc_index in enumerate(y_pred):
            if cc_index < 0 :
                continue
            cc = self.core_concepts[cc_index]
            cc_index_in_terms = self.get_index(cc)
            self.__add_element_to_dict(self.must_link, cc_index_in_terms, term_index)

        return self
    

    def update_cannot_link(self, y_pred : np.ndarray) -> Vocabulary:
        """Modify the cannot link relations based on the prediction
        Associate every predicted hyponym of a core concept to every other core concept and their hyponyms

        Args:
            y_pred (np.ndarray): Result of a prediction 

        Returns:
            Vocabulary: Vocabulary instance
        """
        self.cannot_link = {}
        for term_index, cc_index in enumerate(y_pred):
            if cc_index < 0 :
                continue
            for other_term_index, other_cc_index in enumerate(y_pred):
                if other_cc_index < 0:
                    continue
                if cc_index == other_cc_index :
                    continue
                self.__add_element_to_dict(self.cannot_link, term_index, other_term_index)

        return self