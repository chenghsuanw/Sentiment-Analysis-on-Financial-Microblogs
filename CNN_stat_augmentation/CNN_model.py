import json
import tensorflow as tf
import numpy as np

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

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

    def loadData(self, fileName):
        
        inFile = open(fileName)
        self.postDictList = json.load(inFile)
        self.postNum = len(self.postDictList)

    def loadAugmentedData(self, xFileName, yFileName):

        xListList = np.load(xFileName)
        yListList = np.load(yFileName)

        self.postDictList = []
        for ind in range(len(xListList)):
            postDict = {"snippet": ' '.join(xListList[ind]), "sentiment": yListList[ind][0]}
            self.postDictList += [postDict]

        self.postNum = len(self.postDictList)

        print ('augmented data loaded!')

    def buildDict(self, appearThres):
        
        countDict = {}
        self.wordListList = []
        for postDict in self.postDictList:
            
            snippet = postDict["snippet"]
            wordList = ' '.join(snippet).split(' ') if type(snippet) == list else snippet.split(' ')

            self.wordListList += [wordList]

            for word in wordList:
                lowerWord = word.lower()
                countDict[lowerWord] = countDict.get(lowerWord, 0) + 1
        
        keyList = [key for key in sorted(countDict.keys(), key=lambda x:countDict[x], reverse=1) if countDict[key] >= appearThres]
        self.volNum = len(keyList)

        self.volDict = {}
        for ind in range(self.volNum):
            self.volDict[keyList[ind]] = ind

    def loadDict(self, volNum, volDict, postStdLen):

        self.volNum = volNum
        self.volDict = volDict
        self.postStdLen = postStdLen

        self.wordListList = []
        for postDict in self.postDictList:

            snippet = postDict["snippet"]
            wordList = ' '.join(snippet).split(' ') if type(snippet) == list else snippet.split(' ')
            self.wordListList += [wordList]

    def postToStdLen(self):

        lenList = [len(wordList) for wordList in self.wordListList]
        
        if not self.postStdLen:
            self.postStdLen = max(lenList)
        
        self.yList = []

        for postInd in range(self.postNum):

            self.yList += [[float(self.postDictList[postInd]["sentiment"])]]

            wordIndList = [self.volDict.get(word, self.volNum) for word in self.wordListList[postInd]]
            postLen = lenList[postInd]
            
            repeat = self.postStdLen // postLen
            offset = self.postStdLen % postLen

            self.wordListList[postInd] = wordIndList * repeat + wordIndList[:offset]

    def buildDataDict(self):

        self.dataDict = {"snip": np.array(self.wordListList), "y": np.array(self.yList)}

class Model():

    def __init__(self):
        
        self.model = None
        self.inputPH = None
        self.outputPH = None
        self.lookupTable = None
        self.loss = None
        self.keyPara = None

    def fcLayer(self, inPH, outNodeNum, weightStddev, biasStddev, fcName, fcLayerNum):

        inNodeNum = tf.cast(inPH.shape[1], tf.int32)

        weights = tf.Variable(tf.random_normal([inNodeNum,outNodeNum],stddev=weightStddev),name=fcName+'FcWeight'+str(fcLayerNum))
        bias = tf.Variable(tf.random_normal([outNodeNum], stddev = biasStddev), name = fcName + 'FcBias' + str(fcLayerNum))
        
        return tf.matmul(inPH, weights) + bias

    def convBlock(self, inPH, filtH, filtW, filtNum, weightStdDev, biasStdDev, convName, convLayerNum, maxPoolSize):

        inChaNum = tf.cast(inPH.shape[3], tf.int32)

        randNorm = tf.random_normal([filtH, filtW, inChaNum, filtNum] , stddev = weightStdDev)
        weights = tf.Variable(randNorm, name = convName + 'ConvWeight' + str(convLayerNum))
        bias = tf.Variable(tf.random_normal([filtNum], stddev = biasStdDev), name = convName + 'ConvBias' + str(convLayerNum))

        convPH = tf.nn.conv2d(inPH, weights, strides = [1, 1, 1, 1], padding = 'SAME') + bias
        reluPH = tf.nn.relu(convPH)
        if maxPoolSize != None:
            reluPH = tf.nn.max_pool(reluPH, ksize = [1, 2, 1, 1], strides = [1, 2, 1, 1], padding = 'SAME')

        return reluPH
        
    def buildModel(self, inLen, convFiltNumList, noPoolNum, volEmbedDim, tableStddev, convSize, weightStdDev, biasStdDev, maxPoolSize, fcLayerNum, volNum, keepProb):

        self.keyPara = [convFiltNumList, noPoolNum, volEmbedDim, convSize, fcLayerNum, keepProb]
        convBlockNum = len(convFiltNumList)

        self.inputPH = tf.placeholder('int32', shape = [None, inLen], name = 'inputPH')
        self.yPH = tf.placeholder('float32', shape = [None, 1], name = 'outputPH')

        self.lookupTable = tf.Variable(tf.random_normal([volNum, volEmbedDim], stddev = tableStddev), name = 'lookupTable')
        embedPHPre = tf.nn.embedding_lookup(self.lookupTable, self.inputPH)
        embedPH = tf.expand_dims(embedPHPre, 2)


        # Added for testing.
        #embedPH = tf.nn.dropout(embedPH, keep_prob = keepProb)


        convBlockPH = embedPH
        for blockInd in range(convBlockNum):
            
            if noPoolNum == 0:
                usedMaxPoolSize = maxPoolSize
            else:
                usedMaxPoolSize = None
                noPoolNum -= 1

            filtNum = convFiltNumList[blockInd]
            convBlockPH = self.convBlock(convBlockPH,convSize,1,filtNum,weightStdDev,biasStdDev,'init',blockInd,usedMaxPoolSize)
            convBlockPH = tf.nn.dropout(convBlockPH, keep_prob = keepProb)

        self.fcInputPH = tf.contrib.layers.flatten(convBlockPH)

        fcNodeList = self.getFcNodeList(self.fcInputPH.get_shape().as_list()[1], 1, fcLayerNum)
        
        fcOutputPH = self.fcInputPH
        for fcInd in range(fcLayerNum):
           

            # Added for testing.
            #fcOutputPH = tf.matmul(fcOutputPH,tf.constant([[1/inLen]] * inLen))
            
            
            fcOutputPH = self.fcLayer(fcOutputPH, fcNodeList[fcInd], weightStdDev, biasStdDev, 'final', fcInd)
            if fcInd != fcLayerNum - 1:
                fcOutputPH = tf.nn.dropout(tf.nn.relu(fcOutputPH), keep_prob = keepProb)
            else:
                fcOutputPH = tf.nn.sigmoid(fcOutputPH)

        self.outputPH = fcOutputPH * tf.constant(2.0) - tf.constant(1.0)

        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels = self.yPH, predictions = self.outputPH))

        print ('total parameters: ', self.getTotalParaNum())

    def trainModel(self, learningRate, batchSize, epochNum, feedDictRaw, valFeedDictRaw, printEx):

        optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(self.loss)
        feedDict = {self.inputPH: feedDictRaw['snip'], self.yPH: feedDictRaw['y']}
        valFeedDict = {self.inputPH: valFeedDictRaw['snip'], self.yPH: valFeedDictRaw['y']}

        sampNum = len(feedDictRaw['snip'])
        batchNum = sampNum // batchSize
    
        with tf.Session(config = config) as sess:

            sess.run(tf.global_variables_initializer())
            fileWriter = tf.summary.FileWriter('./summaryGraph', sess.graph)

            for epochInd in range(epochNum):
                print ('epoch ', epochInd + 1)
                print ('convFiltNumList, noPoolNum, volEmbedDim, convSize, fcLayerNum, keepProb: ')
                print (self.keyPara)

                self.shuffleDict(feedDict)

                for batchInd in range(batchNum):

                    #if batchInd % 10 == 0:
                    #    print ('    batch ', batchInd + 1)

                    sess.run(optimizer, feed_dict = self.getBatch(feedDict, batchInd * batchSize, batchSize))
                    
                print ('    training loss:   ', sess.run(self.loss, feed_dict = self.getBatch(feedDict, 0, 2000)))
                print ('    validation loss: ', sess.run(self.loss, feed_dict = valFeedDict))

                if printEx and epochInd % 50 == 0:

                    y_pred = sess.run(self.outputPH, feed_dict = self.getBatch(feedDict, 0, 2000))
                    y_predVal = sess.run(self.outputPH, feed_dict = valFeedDict)
                    y_true = feedDict[self.yPH]
                    y_trueVal = valFeedDict[self.yPH]
                    
                    print ('training pred/true: ')
                    for ind in range(20):
                        print (y_pred[ind], y_true[ind])
                    
                    print ('validation pred/true: ')
                    for ind in range(20):
                        print (y_predVal[ind], y_trueVal[ind])

    def getBatch(self, feedDict, startInd, batchSize):

        batchFeedDict = {}
        for key in feedDict.keys():
            batchFeedDict[key] = feedDict[key][startInd: startInd + batchSize]

        return batchFeedDict

    def shuffleDict(self, feedDict):

        keyList = list(feedDict.keys())

        cummuLenList = [0]
        for key in keyList:
            cummuLenList += [cummuLenList[-1] + len(feedDict[key][0])]

        concatArr = np.concatenate([feedDict[key] for key in keyList], axis = 1)
        np.random.shuffle(concatArr)

        newArrList = np.split(concatArr, cummuLenList[1: -1], axis = 1)

        for ind in range(len(keyList)):
            feedDict[keyList[ind]] = newArrList[ind]

    def getFcNodeList(self, inDim, outDim, layerNum):
        
        growthList = [inDim]
        shrinkList = [outDim]
        
        for ind in range(layerNum-1):

            if growthList[-1] * 2 < shrinkList[-1] * 6:
                growthList += [int(growthList[-1] * 2)]
            else:
                shrinkList += [int(shrinkList[-1] * 6)]
        
        return growthList[1:] + shrinkList[::-1]
        
        #gap = (outDim / inDim) ** (1 / layerNum)
        #return [int(round(inDim * gap ** (ind+1))) for ind in range(layerNum)]

    def getTotalParaNum(self):
        
        totalParaNum = 0
        for variable in tf.trainable_variables():

            print (variable.name, variable.get_shape())
            
            varParaNum = 1
            for dim in variable.get_shape():
                varParaNum *= dim.value
            
            totalParaNum += varParaNum
        
        return totalParaNum

if __name__ == '__main__':
    
    df = DataFormater()
    df.loadData(fileName = "training_set.json")
    #df.loadAugmentedData(xFileName = 'x_list.npy', yFileName = 'y_list.npy')
    df.buildDict(appearThres = 3)
    df.postToStdLen()
    df.buildDataDict()
    
    dfVal = DataFormater()
    dfVal.loadData(fileName = "test_set.json")
    dfVal.loadDict(df.volNum, df.volDict, df.postStdLen)
    dfVal.postToStdLen()
    dfVal.buildDataDict()

    m = Model()
    m.buildModel(inLen = df.postStdLen, convFiltNumList = [4, 8], noPoolNum = 1, volEmbedDim = 2, tableStddev = 0.1, convSize = 3, weightStdDev = 0.01, biasStdDev = 0.01, maxPoolSize = 2, fcLayerNum = 2, volNum = df.volNum + 1, keepProb = 0.8)
    m.trainModel(learningRate = 0.001, batchSize = 1024, epochNum = 10000, feedDictRaw = df.dataDict, valFeedDictRaw = dfVal.dataDict, printEx = 1)
