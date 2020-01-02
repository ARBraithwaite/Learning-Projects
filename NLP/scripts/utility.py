import pandas as pd
import numpy as np

import spacy
import nltk
import re
import unicodedata
import string
from scipy.sparse import csr_matrix

from contractions import CONTRACTION_MAP

nlp = spacy.load("en_core_web_sm") # English Model
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS # stop words

def tokenize(text):
    doc = nlp(text)

    # Create list of word tokens
    token_list = [token.text for token in doc]
    
    return token_list

def sentence_tokenize(text):
    
    doc = nlp(text)
    
    sentences = [sent for sent in doc.sents]
    
    return sentences

def lemmatize(text):
    doc = nlp(text)
    doc = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in doc])
    return doc

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def remove_stop_words(text, stopwords = spacy_stopwords):
    
    doc = tokenize(text)
    
    # filtering stop words
    doc = ' '.join([word for word in doc if word.lower() not in spacy_stopwords])
    
    return doc

def remove_white_space(text):
    doc = tokenize(text)
    
    doc = [word.strip() for word in doc]
    doc = ' '.join([word for word in doc if word != ''])
    
    return doc

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def normalise_text(corpus, All =True, remove_digit=False , expand_contraction=False, remove_special_character=False,\
                   remove_accented_char=False, lemmatise=False, remove_white_spaces=False, remove_stop_word=False,\
                   stopword=spacy_stopwords):
    
    normal_corpus = []
    
    for doc in corpus:
        if All:
            
            text = \
            remove_stop_words(
                remove_white_space(
                    lemmatize(
                        remove_accented_chars(
                            remove_special_characters(
                                expand_contractions(doc), remove_digits=remove_digit
                            )
                        )
                    )
                )
            , stopwords = stopword)
        else:
                
            if expand_contraction:
                text = expand_contractions(doc)
            if remove_special_character:
                text = remove_special_characters(doc, remove_digits=remove_digit)
            if remove_accented_char:
                text = remove_accented_chars(doc)
            if lemmatise:
                text = lemmatize(doc)
            if remove_white_spaces:
                text = remove_white_space(doc)
            if remove_stop_word:
                text = remove_stop_words(doc)
    
        normal_corpus.append(text)
    
    return normal_corpus

def get_entities(corpus):
    named_entities = []
    for doc in corpus:
        
        temp_entity_name = ''
        temp_named_entity = None
        sentence = nlp(doc)

        for word in sentence:
            term = word.text 
            tag = word.ent_type_
            
            if tag:
                temp_entity_name = ' '.join([temp_entity_name, term]).strip()
                temp_named_entity = (temp_entity_name, tag)
                
            else:
                if temp_named_entity:
                    named_entities.append(temp_named_entity)
                    temp_entity_name = ''
                    temp_named_entity = None
                    
    entity_frame = pd.DataFrame(named_entities, 
                                columns=['Entity Name', 'Entity Type'])
    return entity_frame

def count_punct(text):
    doc = tokenize(text)
    
    punctuation = [punct for punct in doc if punct in string.punctuation]
    
    return len(punctuation)

def count_upper(text):
    doc = tokenize(text)
    
    upper  =[word for word in doc if word.isupper()]
    
    return len(upper)

pos_type = {
    'adjective': 'ADJ',
    'adverb': 'ADV',
    'verb': 'VERB',
    'pronoun': 'PRON',
    'noun': 'NOUN',
    
}

def count_pos_type(text, pos: str):
    doc = nlp(text)
    
    if pos in pos_type.keys():
        pos_tagged = [word.pos_ for word in doc if word.pos_ == pos_type[pos]]
        
        return len(pos_tagged)
    else:
        return f'{pos} not in  accepted pos types {pos_type.keys()}'
    

def cooccurrence_matrix(corpus, window_size = 1):
    vocabulary={}
    data=[]
    row=[]
    col=[]
    for sentence in corpus:
        sentence = tokenize(sentence)
        for pos, token in enumerate(sentence):
            i = vocabulary.setdefault(token, len(vocabulary))
            start = max(0, pos-window_size)
            end = min(len(sentence), pos + (window_size + 1))
            for pos2 in range(start, end):
                if pos2 == pos: 
                    continue
                j = vocabulary.setdefault(sentence[pos2],len(vocabulary))
                
                data.append(1.)
                row.append(i)
                col.append(j)
                
    cooccurrence_matrix = csr_matrix((data,(row,col)))
    return vocabulary, cooccurrence_matrix