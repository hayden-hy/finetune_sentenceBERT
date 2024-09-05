from __future__ import annotations
from .Term import Term

class CoreConceptValue:
    """Simple "itérateur" associant un core concept à une valeur numérique
    """
    value = -1
    def __new__(cls):
        cls.value += 1
        return cls.value
    
    @classmethod
    def reset(cls):
        cls.value = -1

class CoreConcept(Term):
    def __init__(self, string: str) -> None:
        super().__init__(string)
        self.sub_concepts = []
        self.cc_value = CoreConceptValue() # if string != "other" else -1
    
    def get_sub_concepts(self) :
        """Méthode renvoyant la liste de tous les hyponymes de ce core-concept, ainsi que lui même
        """
        self.sub_concepts = self.get_hyponyms()
        return self.sub_concepts
    
    def get_core_concept(self):
        """Retourne la valeur associée à un core-concept
        """
        return self.cc_value

    @staticmethod
    def from_term(term : Term):
        instance = CoreConcept(term)
        
        for h in term.hyponyms :
            instance.add_hyponym(h)
        
        instance.set_hypernym(term.hypernym)

        return instance
    