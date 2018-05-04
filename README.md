# Sentiment-Analysis-on-Financial-Microblogs

## How to execute

### Doc2Vec, SVR
```python3.6 word2vec_svr/doc2vec.py```

default the code will execute infer_doc2vec, also infer_word2vec/train_word2vec/train_doc2vec are implemented

### LSTM Method 1 (need tensorflow 1.6 & tqdm)

download all files from google drive (url is in the report)

```cd <directory>``` <br>
and then <br>
```python3 project.py```

run the experiment <br>
```bash exp.sh```

### LSTM-method2

事前處理，下載NTUSD
cd data 
(至data資料夾)

wget http://www.nlg.csie.ntu.edu.tw/nlpresource/NTUSD-Fin/NTUSD-Fin.zip
(下載NTUSD-Fin)

unzip NTUSD-Fin.zip

cd ..
(回到原始資料夾)

training step.
執行python3 rnn.py model_name
(第二個參數為欲儲存model的檔名) 

ex: python3 rnn.py myMODEL
當training結束後，會將該myMODEL.h5儲存於“model”這個資料夾下。



testing step.
執行python3 test.py model_name
(第二個參數為欲讀取model的檔名) 

ex: python3 test.py myMODEL
該指令會到”model”這個資料夾下，抓取myMODEL.h5，作出預測，並顯示MSE loss。
