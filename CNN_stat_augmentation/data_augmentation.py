import numpy as np
import random
import json

class DataAugmentation():

    def __init__(self):

        self.wlNum = None
        self.wlList = None
        self.wlScoreList = None
        self.wlMaxLen = None
        self.wlMinLen = None
        self.cutWlScoreTupList = None
        self.cutWlNum = None
        self.sampListList = None
        self.yListList = None

    def loadData(self, fileName):
        
        inFile = open(fileName)
        postDictList = json.load(inFile)

        snipList = []
        scoreList = []
        for postDict in postDictList:

            snippet = postDict["snippet"]
            score = float(postDict["sentiment"])

            snipList += snippet if type(snippet) == list else [snippet]
            scoreList += [score] * len(snippet) if type(snippet) == list else [score]

        wordListList = []
        wlScoreList = []
        for snipInd in range(len(snipList)):

            snip = snipList[snipInd]
            score = scoreList[snipInd]

            alphaOnly = ''.join([char.lower() for char in snip if (char.isalpha() or char == ' ')])
            wordList = [word for word in alphaOnly.split(' ') if word != '']

            if wordList != []:
                wordListList += [wordList]
                wlScoreList += [score]

        self.wlNum = len(wordListList)
        self.wlList = wordListList
        self.wlScoreList = wlScoreList

        wlLenList = []
        for wl in self.wlList:
            wlLenList += [len(wl)]

#        self.wlMaxLen = max(wlLenList)
#        self.wlMinLen = min(wlLenList)

    def cutSnip(self, cutSize):

        cutWlScoreTupList = []
        for wlInd in range(self.wlNum):

            wl = self.wlList[wlInd]
            score = self.wlScoreList[wlInd]

            length = len(wl)
            startInd = 0
            while startInd < length:
                cutWlScoreTupList += [(wl[startInd: min(startInd + cutSize, length)], score)]
                startInd += cutSize
       
        self.cutWlScoreTupList = sorted(cutWlScoreTupList, key = lambda x: x[1])
        self.cutWlNum = len(cutWlScoreTupList)

        #for tup in self.cutWlScoreTupList:
        #    print (tup)

    def buildData(self, dataLen, classNum, sampNum):

        sampNumPerClass = sampNum // classNum
        poolSizePerClass = self.cutWlNum // classNum

        #print ('sampNum, sampNumPerClass, cutWlNum, poolSizePerClass: ', sampNum, sampNumPerClass, self.cutWlNum, poolSizePerClass)

        sampListList = []
        yListList = []
        for classInd in range(classNum):
            for sampInd in range(sampNumPerClass):

                wl = []
                totScore = 0
                pickNum = 0

                while len(wl) < dataLen:                
                    pickInd = random.randint(classInd * poolSizePerClass, (classInd+1) * poolSizePerClass - 1)
                    tup = self.cutWlScoreTupList[pickInd]
                    
                    wl += tup[0]
                    totScore += tup[1]
                    pickNum += 1
                
                sampListList += [wl]
                yListList += [[totScore / pickNum]]

        self.sampListList = sampListList
        self.yListList = yListList

    def saveData(self, xFileName, yFileName):

        np.save(xFileName, self.sampListList)
        np.save(yFileName, self.yListList)
        print ('files saved!')

    def loadSimulate(self, xFileName, yFileName):

        print ('x: ')
        print (np.load(xFileName))
        print ('y: ')
        print (np.load(yFileName))

if __name__ == '__main__':

    da = DataAugmentation()
    da.loadData(fileName = 'training_set.json')
    da.cutSnip(cutSize = 5)
    da.buildData(dataLen = 25, classNum = 44, sampNum = 100000)
    da.saveData(xFileName = 'x_list.npy', yFileName = 'y_list.npy')
    #da.loadSimulate('x_list.npy', 'y_list.npy')
