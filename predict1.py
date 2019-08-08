import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score

def build_predictions(audio_dir):
    y_true=[]
    y_pred=[]
    fn_prob={}
    list1=[]
    

    print('Extracting..')
    for fn in tqdm(os.listdir(audio_dir)):
        rate,wav=wavfile.read(os.path.join(audio_dir,fn))
        label=fn2class[fn]
        c=classes.index(label)
        y_prob=[]
        for i in range(0,wav.shape[0]-config.step,config.step):
            sample=wav[i:i+config.step]
            x=mfcc(sample,rate,numcep=config.nfeat,nfilt=config.nfilt,nfft=config.nfft)
            x=(x-config.min)/(config.max-config.min)

            x=x.reshape(1,x.shape[0],x.shape[1],1)
            #print('x:',x)
            #print('---------------------------')

            y_hat=model.predict(x)
            list1.append(np.amax(y_hat))

            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
            #print('y',y_hat)
        fn_prob[fn]=np.mean(y_prob,axis=0).flatten()
    return y_true,y_pred, fn_prob,list1
        
df=pd.read_csv('mammals.csv')
true= list(np.array(df.label))


classes= list(np.unique(df.label))
fn2class= dict(zip(df.fname, df.label))
p_path=os.path.join('pickles','conv.p')

with open(p_path,'rb') as handle:
    config=pickle.load(handle)

model=load_model(config.model_path )

y_true,y_pred, fn_prob,list1=build_predictions('clean')

acc_score=accuracy_score(y_true=y_true, y_pred=y_pred)

count=0

y_probs=[]
for i,row in df.iterrows():
    y_prob=fn_prob[row.fname]
    y_probs.append(y_prob)
    for c,p in zip(classes, y_prob):
        df.at[i,c]=p

y_pred=[classes[np.argmax(y)] for y in y_probs]
df['y_pred']= y_pred

df.to_csv('predictions.csv', index=False)

for i in range(0,1225):
    if(np.array(true[i])==np.array(y_pred[i])):
        count+=1
acc=float(count/12.25)
#print('count',count)
print('Accuracy:',acc)



import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
'''for i in range(0,10):
    print('list1',list1[i])'''
print('y t=',y_true)
#print('y pred=',y_pred)

#print('y t len=',len(y_true))

lst=[]
for i in range(0,108356):
    if y_true[i]==2:
        lst.append(1)
    else:
        lst.append(0)
#print('y t==',y_true)
#print('lst==',lst)



fpr, tpr, thresholds = metrics.roc_curve(y_true, lst, pos_label=2)
plt.title('Atlantic spotted dolphin')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
print('tpr:',tpr)
print('fpr:',fpr)
print('threshold:',thresholds)
plt.plot(fpr,tpr)
plt.show()





