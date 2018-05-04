import json
import numpy as np

snippet = []
sentiment = []
true_label = []


with open('training_set.json','r') as f:
	train_list = json.load(f)
	for index,dic in enumerate(train_list):
		if type(dic["snippet"])  == type("string"):
			snippet.append(dic["snippet"])
			sentiment.append(dic["sentiment"])
		else: #snippet is a list of string
			for string in dic["snippet"]:
				#print(string)
				snippet.append(string)
				sentiment.append(dic["sentiment"])
		'''
		if int(dic["sentiment"]) > 0:
			true_label.append(1)
		elif int(dic["sentiment"]) < 0:
			true_label.append(-1)
		else:
			true_label.append(0)
		'''
			#print()

np.save("data/snippet",snippet)
np.save("data/sentiment",sentiment)


#np.save("data/true_label",true_label)

###check###
#for i in range(20):
#	print(snippet[i],sentiment[i])


snippet = []
sentiment = []
true_lable = []

with open('test_set.json','r') as f:
	test_list = json.load(f)
	for index,dic in enumerate(test_list):
		if type(dic["snippet"])  == type("string"):
			snippet.append(dic["snippet"])
			sentiment.append(dic["sentiment"])
		else: #snippet is a list of string
			concatenate = ""
			for string in dic["snippet"]:
				#print(string)
				concatenate += string
			snippet.append(concatenate)
			sentiment.append(dic["sentiment"])
		'''
		if int(dic["sentiment"]) > 0:
			true_label.appned(1)
		elif int(dic["sentiment"]) < 0:
			true_label.append(-1)
		else:
			true_label.append(0)
			#print()
		'''	
np.save("data/Test_snippet",snippet)
np.save("data/Test_sentiment",sentiment)
#np.save("data/Test_true_label",true_label)

#print(len(snippet))
#print(len(sentiment))
