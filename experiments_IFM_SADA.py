import numpy as np
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from DANN_IFM_SADA import DANN
import scipy.io
from matplotlib import pyplot
from scipy.spatial.distance import cdist

dataDir='./data_dataset/'
file = scipy.io.loadmat(dataDir+'IFM_KERCOV_PCA1000.mat')
np.random.seed(4)
data=file['data']
np.random.shuffle(data)
subj = data[:,0]-1
act = data[:,1]-1
data = data[:,2:]
lam=[0.1]
accuracy_NN=np.zeros(17)
accuracy_DANN=np.zeros(17)
for ii in range(len(lam)):
  for sogg in range(17):
    s=sogg #test subject
    num_actions = len(np.unique(act))
    X_train = data[subj!=s]
    X_test= data[subj==s]
    Y_train = act[subj!=s]
    Y_test= act[subj==s]
    subj_train = subj[subj!=s]
    subj_test = subj[subj==s]
    X=X_train
    y=Y_train
    Xt=X_test
    yt=Y_test
    y_domain = subj_train
    yt_domain = subj_test 
    print('SUBJ ' + str(sogg) + '\n')
    algo = DANN(hidden_layer_size=500, maxiter=700, lambda_adapt=lam[ii], seed=42)
    algo.fit(X, y, y_domain, Xt,yt,yt_domain)
    y_pred_DANN=algo.predict(Xt)
    print('Acc:',100*np.sum(y_pred_DANN==yt)/float(len(yt)), '\n')
 
  
