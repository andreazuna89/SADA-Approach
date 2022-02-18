import sys
import os
import matplotlib.pyplot as plt
import numpy as np

test_acc=[]
test_loss = []
train_acc=[]
train_loss = []
testX=[]
testCntr=0
subj=0
test=0
train=0
f = open('./res_acc'+str(subj)+'.txt', 'r')
#f = open('./res_loss'+str(subj)+'.txt', 'r')
idx=0
for line in f:
        idx+=1
        #pdb.set_trace()
        train+=float(line.split('\t')[0])
        test+=float(line.split('\t')[1][:-1])
        if idx%1098==0 & idx !=1:
          test_acc.append( test/1098)
          testCntr =  testCntr + 1
          train_acc.append( train/1098)
        
          testX.append(testCntr)
          train=0
          test=0
f.close()
test_acc=np.array(test_acc)
train_acc=np.array(train_acc)
testX=np.array(testX)
plt.figure(figsize=(10,6))
plt.plot(testX,test_acc,'g.-',label='Test Loss')
plt.plot(testX,train_acc,'y.-',label='Train Loss')
plt.xlabel('Epoch')
#plt.ylabel('Value')
plt.title('SADA Model accuracy for test subject '+str(subj+1))
#plt.title('SADA Model losses for test subject '+str(subj+1))
#plt.ylim([0.4,0.6])
#box = plt.get_position()
#plt.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='best')
plt.grid(True)
plt.show()
