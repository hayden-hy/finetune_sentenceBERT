from __future__ import annotations

import random

import pandas as pd
import spacy
import numpy
import json
# import gpl

from nltk.corpus import stopwords
from textblob import TextBlob
from textblob.blob import Word
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from tools.HearstPatterns import HearstPatterns
from tools.pdfToCleanText import pdfToCleanText

import warnings

from .Terms import *


class Corpus() :
    def __init__(
        self,
        source          : str                   = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTSsp1oV819VYKVBC8JV3Cbsat2Q9iEL0Zh_-RMIRgrP3eR9RkWLceBVrzlBrEPlZ9sMnBROvKWo8Hm/pub?gid=1982298826&single=true&output=csv",
        columns         : list[str]             = ["Concept","Core Concept"],
        stop_words      : list[str]             = stopwords.words(), 
        model           : SentenceTransformer   = SentenceTransformer('all-MiniLM-L6-v2'),
        core_concepts   : list[str]             = None,
        prefix          : str                   = ""
    ) -> None:
        """
        Initializes an instance of the Corpus class. Sets initial values for ontology, columns, sentences, vocabulary, stop_words, the SentenceTransformer model, and a boolean flag (`old`).

        Parameters:
            source (str): URL of the CSV file to be used as the source for ontology.
            columns (list[str]): List of column names to be used from the CSV file. Default is ["Concept", "Core Concept"].
            stop_words (list[str]): List of stop words to be ignored in the text. Default is NLTK's list of English stop words.
            model (SentenceTransformer): SentenceTransformer model to be used for embeddings. Default is 'all-MiniLM-L6-v2'.
            old (bool): Boolean flag to specify whether to use old core concepts. Default is False.

        Returns:
            None
        """
        file_extension = source.split(".")[-1]
        self.ontology                               = pd.read_excel(source) if file_extension == "xlsx" else pd.read_csv(source)
        self.colums         : list[str]             = columns

        self.ontology                               = self.ontology[columns].dropna() 
        self.ontology[columns[0]]=self.ontology[columns[0]].str.lower() # lower case all terms
        self.ontology[columns[1]]=self.ontology[columns[1]].str.lower()
        
        self.sentences      : list[str]             = []
        self.vocabulary     : Vocabulary            = Vocabulary()
        self.stop_words     : list[str]             = stop_words
        self.model          : SentenceTransformer   = model

        self.core_concepts = core_concepts
        # replace "Acrtifiial Intelligence" by "AI" in the ontology
        self.ontology[columns[1]] = self.ontology[columns[1]].apply(lambda x: "artificial intelligence" if x == "anythingai" else x)
        self._add_prefix(prefix=prefix.lower(),column=columns[1])

        if self.core_concepts != None :
            self._parse_ontology(self.core_concepts)

        print(self.vocabulary.terms)
        self._get_core_concepts_from_ontology(column=columns[1])

        print(self.vocabulary.core_concepts)


    def _add_term(self, term : Term):
        """
        Appends a Term object to the vocabulary.

        Parameters:
            term (Term): The Term object to be appended.

        Returns:
            None
        """
        self.vocabulary.append(term)

    
    def _get_term_from_string(self, string : str) -> Term:
        """
        Returns a Term object that matches a given string from the vocabulary.

        Parameters:
            string (str): The string to match in the vocabulary.

        Returns:
            Term: The matching Term object.
        """
        return self.vocabulary.get_term_from_string(string)


    def _get_core_concepts_from_ontology(self, column="Core Concept"):
        """
        Iterates over core concepts in the ontology, adds them to the vocabulary, and links them to related terms.

        Parameters:
            column (str): The column in the ontology DataFrame that contains core concepts. Default is "Core Concept".

        Returns:
            None
        """
        core_concepts = pd.unique(self.ontology[column])
       
        for cc in core_concepts :
            cc = cc.lower()

            if cc not in self.vocabulary.terms :
                warnings.warn("Core concept \"" + cc + "\" not found in manual ontology")

            cc = CoreConcept.from_term(self.vocabulary.get_term_from_string(cc))

            for hyponym in cc.hyponyms :
                self.__get_core_concept_hyponyms(cc, hyponym)

            self.vocabulary.core_concepts.append(cc)
            term_index = self.vocabulary.get_index(self.vocabulary.get_term_from_string(cc))
            self.vocabulary.terms[term_index] = cc

            for other_cc in self.vocabulary.core_concepts :
                if other_cc == cc :
                    continue
                self.vocabulary.add_cannot_link(cc, other_cc)


    def __get_core_concept_hyponyms(self, core_concept : CoreConcept, term : Term):
        """
        Recursively links a core concept with its hyponyms in the vocabulary.

        Parameters:
            core_concept (CoreConcept): The core concept to be linked.
            term (Term): The term to be linked as a hyponym of the core concept.

        Returns:
            None
        """
        self.vocabulary.add_must_link(core_concept, term)

        if len(term.hyponyms) == 0:
            return
        
        for hyponym in term.hyponyms :
            self.__get_core_concept_hyponyms(core_concept, hyponym)


    def _get_fine_tune_training_data(self) :
        """
        Generates training data for fine-tuning the SentenceTransformer model using Next Sentence Prediction.

        Returns:
            list: A list of training data instances (InputExample objects).
        """
        num_sentences = len(self.sentences)
        sentence_a = ""
        sentence_b = ""
        label = 0

        training_data = []

        # Entrainement de BERT sur de la Next Sentence Prediction pour fine tune l'embedding
        for i in range(nb_training_data):
            random_index = random.randint(0, num_sentences - 2)

            # Select 50/50 wether the pair will be correct or not
            if random.random() >= 0.5 :
                # NextSentence will be correct
                sentence_a = self.sentences[random_index]
                sentence_b = self.sentences[random_index + 1]
                label = 0.0
            else :
                other_index = random.randint(0, num_sentences - 1)
                if (other_index == random_index + 1):
                    other_index -= 1
                sentence_a = self.sentences[random_index]
                sentence_b = self.sentences[other_index]
                label = 1.0

            training_data.append(InputExample(texts=[sentence_a, sentence_b], label=label))
        return training_data
        

    def fine_tune_model(self, nb_training_data = 500, epochs = 1, warmup_steps = 100) :
        """
        Fine-tunes the SentenceTransformer model using the generated training data.

        Parameters:
            nb_training_data (int): The number of training data samples to be used for fine-tuning. Default is 500.
            epochs (int): The number of epochs to use for training. Default is 1.
            warmup_steps (int): The number of steps for the warmup phase. Default is 100.

        Returns:
            None
        """
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(self.model)

        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=warmup_steps)
        self.model.save("./Model", "test")
        

    def extract_from_pdf(self, pdf_path) -> Corpus:
        """
        Extracts sentences from a given PDF and adds them to the sentences attribute.

        Parameters:
            pdf_path (str): The path to the PDF file.

        Returns:
            Corpus: The current Corpus instance.
        """
        self.sentences = [" ".join(sentence) for sentence in pdfToCleanText(pdf_path)]
        return self
    
    
    def extract_terms(self) -> Corpus:
        """
        Uses the spaCy library to extract terms from sentences and adds them to the vocabulary.

        Returns:
            Corpus: The current Corpus instance.
        """
        nlp = spacy.load("en_core_web_sm")

        for sentence in self.sentences:
            doc = nlp(sentence)
            for s in doc.sents :
                for chunk in s.noun_chunks :
                    term = self._get_term_without_stopwords(chunk)

                    nb_words = len(term.split(" "))
                    if nb_words > 3:
                        continue

                    if term in self.vocabulary.terms_idx :
                        continue

                    if term == "":
                        continue

                    self._add_term(term)
        return self
    
    def extract_terms_new(self) -> Corpus :
        """
        Uses the TextBlob library to extract noun phrases from sentences, lemmatizes them, and adds them to the vocabulary.

        Returns:
            Corpus: The current Corpus instance.
        """
        blob = TextBlob(" ".join(self.sentences))
        for noun_phrase in blob.noun_phrases :
            noun_phrase = noun_phrase
            lemmas = []
            for word in noun_phrase.split(" "):
                lemmas.append(Word(word).lemmatize())
            noun_phrase_lemma = " ".join(lemmas)
            self._add_term(Term(noun_phrase_lemma))
        return self
    
    def extract_terms_from_onto(self) :
        """
        Extracts unique terms from the ontology and adds them to the vocabulary.

        Returns:
            None
        """
        terms = pd.unique(self.ontology[self.colums[0]])

        dup_terms = self.ontology[self.colums[0]][self.ontology[self.colums[0]].duplicated()]
        if len(dup_terms) != 0:
            print("redundant terms:")
            print(dup_terms)

        print("unique terms: ", len(terms))

        for term in terms :
            self._add_term(Term(term.lower()))
    
    def extract_terms_from_pdf(self, pdf_path):
        """
        Extracts terms from a given PDF and adds them to the vocabulary.

        Parameters:
            pdf_path (str): The path to the PDF file.

        Returns:
            Corpus: The current Corpus instance.
        """
        self.extract_from_pdf(pdf_path).extract_terms_new()
        return self
    
    def extract_hyponyms(self):
        """
        Extracts hyponyms from terms using the Hearst Patterns method.

        Returns:
            None
        """
        # self.__get_hyponyms_from_text()
        self.__get_hyponyms_from_term_head()


    def __get_hyponyms_from_text(self):
        """
        Extracts hyponyms from sentences using the Hearst Patterns method.

        Returns:
            None
        """
        h = HearstPatterns(extended = True)

        hyponyms = []

        for sentence in self.sentences :
            hyponyms += h.find_hyponyms(sentence)

        for hyponym, hypernym in hyponyms:
            if hypernym not in self.vocabulary.terms_idx or hyponym not in self.vocabulary.terms_idx:
                continue
            
            hypernym = self._get_term_from_string(hypernym)
            hyponym = self._get_term_from_string(hyponym)

            self.__set_hypernym_relation(hypernym, hyponym)


    def __get_hyponyms_from_term_head(self):
        """
        Extracts hyponyms from the head of terms.

        Returns:
            None
        """
        for term in self.vocabulary :
            head = term.head
            if head == None:
                continue

            if head not in self.vocabulary.terms_idx :
                continue

            hypernym = self._get_term_from_string(head)
            self.__set_hypernym_relation(hypernym, term)
    

    def __set_hypernym_relation(self, hypernym : Term, hyponym : Term) -> Corpus:
        """
        Sets hypernym-hyponym relations in the vocabulary.

        Parameters:
            hypernym (Term): The Term object to be set as a hypernym.
            hyponym (Term): The Term object to be set as a hyponym.

        Returns:
            Corpus: The current Corpus instance.
        """
        if hypernym.get_core_concept() != -1 :
            self.vocabulary.add_must_link(hypernym, hyponym)
            print(hyponym, "-->" , self.vocabulary.core_concepts[hypernym.get_core_concept()])

        print(hyponym, "-->" , hypernym)

        hypernym.add_hyponym(hyponym)
        return self


    def update_vocabulary(self, y_pred):
        """
        Updates the vocabulary based on predicted labels.

        Parameters:
            y_pred: Predicted labels.

        Returns:
            None
        """
        self.vocabulary.update_vocabulary(y_pred)


    def _get_term_without_stopwords(self, noun_chunk):
        """
        Returns a Term object from a given noun chunk with stop words removed.

        Parameters:
            noun_chunk: The noun chunk from which the term is to be created.

        Returns:
            Term: The Term object created from the noun chunk.
        """
        term = []
        for token in noun_chunk :
            if token.lemma_ in self.stop_words :
                continue

            if token.lemma_.isalnum():
                term.append(token.lemma_)
            else:
                term.append(''.join(
                    char for char in token.lemma_ if char.isalnum()
                ))
        term = Term(" ".join(term).lower())
        term.head = noun_chunk.root
        return term
    
    def _get_embedding(self, term) :
        """
        Returns the embedding of a given term.

        Parameters:
            term: The term to encode.

        Returns:
            The embedding of the term.
        """
        return self.model.encode(term)

    def get_training_data(self):
        """
        Returns training data (term embeddings and associated core concepts) for a model.

        Returns:
            numpy array: An array of term embeddings (X) and associated core concepts (y).
        """
        X = []
        y = []

        for term in self.vocabulary :
            X.append(self._get_embedding(term))
            y.append(term.get_core_concept())

        return numpy.array(X), numpy.array(y)
    

    def _parse_ontology(self, ontology, hypernym : Term | None = None):
        """
        Recursively parses an ontology, creating and adding Term objects to the vocabulary.

        Parameters:
            ontology: The ontology to parse.
            hypernym (Term): The hypernym for the terms in the ontology. Default is None.

        Returns:
            None
        """
        if len(ontology.items()) == 0:
            return

        
        for cc, sub_concepts in ontology.items() :
            term = Term(cc)
            self.vocabulary.append(term)

            if hypernym != None :
                hypernym.add_hyponym(term)
            
            self._parse_ontology(sub_concepts, term)

    def _add_prefix(self, prefix : str,column : str) :
        """
        Adds a prefix to all terms in the vocabulary.
        
        Parameters:
            prefix (str): The prefix to add on core concepts.
            column (str): The column in the ontology DataFrame that contains core concepts.
        """

        # domain corpus
        self.ontology[column] = self.ontology[column].apply(lambda x: prefix + x if x != "other" else x)
        # core concepts
        temp_dict = {}
        for key, value in self.core_concepts.items():
            if key != "other":
                temp_dict[prefix + key] = value
            else:
                temp_dict[key] = value
        self.core_concepts = temp_dict