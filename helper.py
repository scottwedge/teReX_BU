import os
import pdb

def createConfigName(cosSim, featureConv):
	name='_'
	if cosSim:
		name = name + 'cos'
	name = name + '_' + str(featureConv) 
	return name 


def createDirectory(path):
	if not os.path.exists(path):
		os.makedirs(path)

def generateVocabulary(data):
	sentences = flattenList(data)
	vocabularySet = getWordSetOfList(sentences)
        vocabList = list(vocabularySet)
        vocabList.sort()
	return createDictionary(vocabList)

def getWordSetOfList(wordList):
	wordSet = set()
	for words in wordList:
		wordSet.update(words)
	return wordSet
		

def flattenList(multiLevelList):
	return [sum(elem, []) for elem in multiLevelList]

def createDictionary(List):
	mapping = zip(List, range(len(List)))
	return dict(mapping)
	
