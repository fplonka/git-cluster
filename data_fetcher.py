import scipy.stats
import numpy as np

# fetches a dataset and target values. be sure to have X values encoded in np.float64
def fetch_dataset(label):
    if label == 'coil20':
        getter = get_coil20
    elif label == 'airfoil':
        getter = get_airfoil_noise
    elif label == 'wine':
        getter = get_winequality
    elif label == 'RNAseq':
        getter = get_RNAseq_3k
    else:
        print("unrecognised dataset name: \'", label, "\' \n\n")
        1/0
    X, Y = getter()
    print('loaded '+label+' : (N, M) = '+str(X.shape), '\n')
    return X.astype(np.float64), Y

def get_coil20():
    from scipy.io import loadmat
    mat = loadmat("datasets/COIL20.mat")
    X, Y = mat['X']+1e-8, mat['Y']
    Y = (Y.astype(int) - 1).reshape((-1,))
    return X, Y

def get_airfoil_noise():
    label_idx = -1
    XY = np.genfromtxt('datasets/airfoil_noise.csv', delimiter=";", skip_header=1)
    Y = XY[:, label_idx]
    X = np.delete(XY, label_idx, axis=1)
    return scipy.stats.zscore(X, axis=0), Y

def get_RNAseq_3k():
    XY = np.load('datasets/RNAseq_N3k.npy')
    return XY[:, :-1], XY[:, -1]

def get_winequality():
    XY = np.genfromtxt('datasets/winequality-red.csv', delimiter=";", skip_header=1)
    Y = XY[:, -1]
    X = XY[:, :-1]
    return scipy.stats.zscore(X, axis=0), Y
