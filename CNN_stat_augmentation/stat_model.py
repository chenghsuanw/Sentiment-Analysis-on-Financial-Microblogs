import json
import numpy as np

class DataFormater():

    def __init__(self):

        self.postDictList = None
        self.volNum = None
        self.volDict = None
        self.dataDict = None
        self.wordListList = None
        self.postNum = None
        self.postStdLen = None
        self.yList = None
        self.scoreList = None
        self.scoreDict = None

    def loadData(self, fileName):

        inFile = open(fileName)
        self.postDictList = json.load(inFile)
        self.postNum = len(self.postDictList)

    def buildWordListList(self):
        
        self.wordListList = []
        self.scoreList = []
        for postDict in self.postDictList:
            
            snippet = postDict["snippet"]
            score = float(postDict["sentiment"])
            wordList = ' '.join(snippet).split(' ') if type(snippet) == list else snippet.split(' ')
            
            alphaWordList = []
            for word in wordList:
                
                alphaWord = ''
                for char in word:
                    alphaWord += char.lower() if char.isalpha() else ''

                if alphaWord != '':
                    alphaWordList += [alphaWord]

            if alphaWordList != []:
                self.wordListList += [alphaWordList]
                self.scoreList += [score]

        #for wl in self.wordListList:
        #    print (wl)

    def buildScoreDict(self):

        self.scoreDict = {}
        for wlInd in range(len(self.wordListList)):

            wl = self.wordListList[wlInd]
            score = self.scoreList[wlInd]

            for word in wl:
                self.scoreDict[word] = self.scoreDict.get(word, []) + [score]

        self.countDict = {}
        for key in self.scoreDict.keys():
            self.countDict[key] = len(self.scoreDict[key])
            self.scoreDict[key] = np.mean(self.scoreDict[key])

        tupList = sorted([(key, self.countDict[key], self.scoreDict[key]) for key in self.scoreDict.keys()], key = lambda x:x[2])
        
        #for tup in tupList:
        #    print (tup)

    def getLoss(self, countDict, scoreDict, countThres, countMax, idfEp):
        
        totalLoss = 0
        sampNum = len(self.wordListList)
        predList = []

        truePos = 0
        trueNeg = 0
        falsePos = 0
        falseNeg = 0

        for sampInd in range(sampNum):
            wl = self.wordListList[sampInd]

            if not idfEp:

                wordScoreList = [scoreDict.get(word, 0) for word in wl if (countDict.get(word, 0) >= countThres and countDict.get(word, 0) <= countMax)]
                
                predScore = 0 if not wordScoreList else np.mean(wordScoreList) 

            else:

                wordScoreList = [scoreDict.get(word, 0) / (countDict[word] + idfEp) for word in wl if (countDict.get(word, 0) >= countThres and countDict.get(word, 0) <= countMax)]
                
                wordWeightList = [np.log(2) / np.log(countDict[word] + idfEp) for word in wl if (countDict.get(word, 0) >= countThres and countDict.get(word, 0) <= countMax)]
                
                predScore = 0 if not wordScoreList else np.sum(wordScoreList) / np.sum(wordWeightList)
            
            predList += [predScore]
            totalLoss += (predScore - self.scoreList[sampInd]) ** 2

            if predScore > 0:
                if self.scoreList[sampInd] > 0:
                    truePos += 1
                elif self.scoreList[sampInd] < 0:
                    falsePos += 1
            elif predScore < 0:
                if self.scoreList[sampInd] > 0:
                    falseNeg += 1
                elif self.scoreList[sampInd] < 0:
                    trueNeg += 1

        maPre = (truePos / (truePos + falsePos) + trueNeg / (trueNeg + falseNeg)) / 2
        maRec = (truePos / (truePos + falseNeg) + trueNeg / (trueNeg + falsePos)) / 2
        miPre = (truePos + trueNeg) / (truePos + falsePos + trueNeg + falseNeg)
        miRec = (truePos + trueNeg) / (truePos + falseNeg + trueNeg + falsePos)

        print ('macro f1: ', 2 / (1 / maPre + 1 / maRec))
        print ('micro f1: ', 2 / (1 / miPre + 1 / miRec))

        mse = totalLoss / sampNum
        return mse

if __name__ == '__main__':
    
    df = DataFormater()
    df.loadData(fileName = 'training_set.json')
    df.buildWordListList()
    df.buildScoreDict()

    dfVal = DataFormater()
    dfVal.loadData(fileName = 'test_set.json')
    dfVal.buildWordListList()
    dfVal.buildScoreDict()
    mse = dfVal.getLoss(df.countDict, df.scoreDict, 1, 90, idfEp = None)
    print ('mse: ', mse)
