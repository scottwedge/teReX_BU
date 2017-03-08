from GraphDatabase import GraphDatabase
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import sent_tokenize
from sklearn.datasets import fetch_20newsgroups
import pandas as pd 
import numpy as np
from numpy import transpose, identity
from preprocessing import lemmatizeAll, tokenizeSentence
import os.path
from helper import generateVocabulary
import pdb

def script():
	
	database  = GraphDatabase()
	name = 'NG_rec_test'
	filename = '/../vol/tensusers/nwidmann/processedDocuments/'+ name +'.pkl'
	minFrequency = 10 
	pdb.set_trace()

	if not os.path.exists(filename):
        	print 'Load Documents'
		data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
		pdb.set_trace()
		#data = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'], remove=('headers', 'footers', 'quotes'))
        	#data = fetch_20newsgroups(categories=['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'], remove=('headers', 'footers', 'quotes'))
		#data = fetch_20newsgroups(categories=['talk.politics.guns', 'rec.motorcycles'], remove=('headers', 'footers', 'quotes'))
		categories = data.target_names
		data = pd.DataFrame({'text': data['data'], 'category': data['target']})
		#data = data[0:100]

        	for index, category in enumerate(categories):
			print 'Category: ' + category + '   N: ' + str(len(data[data.category==index]))

        	print 'Preprocessing'
        	docs = data.text.tolist()
        	vectorizer = CountVectorizer(min_df=minFrequency, stop_words='english', token_pattern='[a-zA-Z]+', max_df=0.5)
		#vectorizer = CountVectorizer(min_df=minFrequency, token_pattern='[a-zA-Z]+')
        	wordCounts = vectorizer.fit_transform(docs)
        	vocabulary = vectorizer.get_feature_names()
        	print('Number of Unique words: %d' % len(vocabulary))
        	print('Minimal Frequency: %d' % minFrequency)

		docsSplitInSentences = [sent_tokenize(doc) for doc in docs]
		tokenizedCollection = [[tokenizeSentence(sentence) for sentence in sentences] for sentences in docsSplitInSentences]

		cleanedTokens = [[[lemmatizeAll(word.lower()) for word in sentence if word.lower() in vocabulary and len(word)>1] for sentence in doc] for doc in tokenizedCollection]
		cleanedTokens = [filter(None, doc) for doc in cleanedTokens]
		data['sentences'] = cleanedTokens
		vocabulary = generateVocabulary(data.sentences.tolist())
		
		fullCleanText = [' '.join(sum(post, [])) for post in data.sentences.tolist()]
		data['cleanText'] = fullCleanText

		tfIdf = TfidfVectorizer(vocabulary=vocabulary)
		docs = data.cleanText.tolist()
		tfidf_vec = tfIdf.fit_transform(docs)
		data['tfIdf'] = [list(elem) for elem in tfidf_vec.toarray()]

		tf = CountVectorizer(vocabulary=vocabulary)
		tf_vec = tf.fit_transform(docs)
		data['tf'] = [list(elem) for elem in tf_vec.toarray()]

		# Remove posts with no features
		for index in range(len(data)):
			tfIdfSum = np.sum(data.loc[index, 'tfIdf'])
			if tfIdfSum==0:
				print index
				data.drop(index, inplace=True)
		data.index = range(len(data))
			
        	data.to_pickle(filename)
	
	data = pd.read_pickle(filename)
	vocabulary = generateVocabulary(data.sentences.tolist())
	data.sentences = data.sentences[0:70]
	#pdb.set_trace()

	#toydata = [[0, [['This','is','it','.'],['it','.']]], [1,[['it','is','here','is','.']]]]
	#data = pd.DataFrame(toydata, columns=['category', 'sentences'])

	print 'Graph Construction'
	startNode = database.createFeatureNode(-1,'$Start$')
	endNode = database.createFeatureNode(len(vocabulary), '$End$')
	for index, text in enumerate(data.sentences):
		print 'Document' + str(index)
		label = data.category.loc[index]
		docNode = database.createDocumentNode(index, label)
		for sentence in text:
			preceedingWord = startNode
			database.createWeightedRelation(startNode,docNode, 'is_in')
			for ind,word in enumerate(sentence):
				exists = len(list(database.graph.find('Feature', property_key='word', property_value=word))) > 0
				if not exists:
					wordID = vocabulary[word]
					wordNode = database.createFeatureNode(wordID, word)
				else:
					wordNode = database.getFeatureNode(word)
				database.createWeightedRelation(wordNode, docNode, 'is_in')
				database.createWeightedRelation(preceedingWord, wordNode, 'followed_by')
				preceedingWord = wordNode
				if ind==len(sentence)-1:
					database.createWeightedRelation(wordNode, endNode, 'followed_by')
					database.createWeightedRelation(endNode, docNode, 'is_in')

	print 'Normalize relationships'
	docNodes = database.getNodes('Document')
	database.normalizeRelationships(docNodes, 'is_in')
	
	featureNodes = database.getNodes('Feature')
	database.normalizeRelationships(featureNodes, 'followed_by')

	print 'Create Matrix'
	#docMatrix = identity(len(docNodes))
	#featureMatrix = database.getMatrix(featureNodes)
	#featureDocMatrix = database.getMatrix(featureNodes, docNodes, 'is_in')
	#docAll = np.concatenate((docMatrix, np.transpose(featureDocMatrix)), axis=1)
	#featureAll = np.concatenate((featureDocMatrix, featureMatrix), axis=1)
	#combinedMatrix = np.concatenate((docAll, featureAll))
	#print combinedMatrix.shape
	#np.save('/../vol/tensusers/nwidmann/matrices/' + name, combinedMatrix)


	print 'Set Context Similarity'
	database.cypherContextSim()
	contextSim = database.getMatrix(featureNodes, relation='related_to', propertyType = 'contextSim')
	np.save('/../vol/tensusers/nwidmann/matrices/' + name + '_contextSim', contextSim)


if __name__ == '__main__':
    script()
