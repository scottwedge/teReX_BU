#coding=utf-8
from sklearn.feature_extraction import stop_words
import pandas as pd
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

stopwords = set(stop_words.ENGLISH_STOP_WORDS)
stopwords.update([',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])     

WordNet = WordNetLemmatizer()
regex = re.compile('[a-zA-Z]+')

def split(text):
    return [word_tokenize(sentence.strip()) for sentence in text]

def lemmatize(tokens):
    return [WordNet.lemmatize(WordNet.lemmatize(word), 'v') for word in tokens]

def tokenizeSentence(sentence):
    return regex.findall(sentence)

def lemmatizeAll(word):
    return WordNet.lemmatize(WordNet.lemmatize(WordNet.lemmatize(word), 'v'), 'a')

def lowerAndTokenize(text):
    return word_tokenize(text.lower())

def removeStopwords(tokens):
    return [word for word in tokens if word not in stopwords]

def standardPreprocessing(data, filename):
    data['tokens'] = data['text'].apply(lowerAndTokenize) 
    
    data['cleanText'] = data['tokens'].apply(lemmatize) 
    data['cleanText'] = data['cleanText'].apply(removeStopwords) 
    data['cleanText'] = data['cleanText'].apply(' '.join)
                                                                  
    data['sentences'] = data['cleanText'].apply(sent_tokenize)
    data['sentences'] = data['sentences'].apply(split)
                                                      
    data.to_pickle(filename)


def tokenize(text):
	text = text.lower()
	text = re.sub("\n", " ", text)
	text = re.sub(r'<[^>]+>', " ", text)
	text = re.sub('[^a-zèéeêëėęûüùúūôöòóõœøîïíīįìàáâäæãåçćč&@#A-ZÇĆČÉÈÊËĒĘÛÜÙÚŪÔÖÒÓŒØŌÕÎÏÍĪĮÌ0-9- \']', "", text)
	words = text.split()
	words = [word for word in words if len(word)>1 and len(word)<20]
	return words

def getStopwords():
	stopwords = set()
        with open('stoplist.txt') as f:
        	for line in f:
        		stopwords.add(line.rstrip())
	return stopwords
