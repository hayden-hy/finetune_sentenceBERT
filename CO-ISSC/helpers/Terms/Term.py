from __future__ import annotations

class Term(str):
    def __new__(cls, *args, **kwargs) -> Term:
        """Méthode pour override la classe str
        """
        instance = super(Term, cls).__new__(cls, *args, **kwargs)
        instance._from_base_class = type(instance) == Term
        return instance

    def __init__(self, string : str) -> None:
        self.hyponyms = []
        self.hypernym = None
        self.head = None

    def add_hyponym(self, hyponym : str | Term) -> Term:
        """Ajoute la relation d'hyponymie entre deux termes
        """
        if (type(hyponym) == str) :
            self.add_hyponym(Term(hyponym))
            return self
        
        # On définit également la relation d'hyperonymie
        # pour l'autre terme
        hyponym.set_hypernym(self)
        self.hyponyms.append(hyponym)

        return self
    
    def set_hypernym(self, hypernym : Term) -> Term :
        self.hypernym = hypernym
        return self

    def get_hyponyms(self) :
        """Méthode retournant récursivement tous les hyponymes d'un terme
        """
        hyponyms = [self]
        for hyponym in self.hyponyms :
            hyponyms += hyponym.get_hyponyms()
        return hyponyms
    
    def get_core_concept(self) :
        """Retourne la valeur associée à un core-concept
        Si le terme n'est pas un core concept, et qu'il n'a pas d'hyperonyme, 
        renvoie -1 pour signifier qu'il n'est associé à aucun core-concept
        Cette fonction est override par la classe CoreConcept
        """
        if self.hypernym == None :
            return -1
        
        return self.hypernym.get_core_concept()
