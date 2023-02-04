import pickle
from scipy.io import loadmat
import numpy as np
import scipy as sp 
import scipy.linalg as la
import sklearn.discriminant_analysis as lda

def A3Load():
    # Set A3 directory
    DIR = './'

    # Load the train data and format it
    EEGL_train = pickle.load(open(DIR+'data/tr_EEGL_1.pkl', "rb"))['L'] # (9, 11)
    EEGR_train = pickle.load(open(DIR+'data/tr_EEGR_1.pkl', "rb"))['R'] # (9, 11)
    for i in range(2,10):
        EEGL_train = np.vstack((EEGL_train,pickle.load(open(DIR+'data/tr_EEGL_'+str(i)+'.pkl', "rb"))['L']))
        EEGR_train = np.vstack((EEGR_train,pickle.load(open(DIR+'data/tr_EEGR_'+str(i)+'.pkl', "rb"))['R']))

    # Load labeled eval data and format it
    EEGL_eval = pickle.load(open(DIR+'data/ev_EEGL_1.pkl', "rb"))['L'] # (9, 11)
    EEGR_eval = pickle.load(open(DIR+'data/ev_EEGR_1.pkl', "rb"))['R'] # (9, 11)
    for i in range(2,10):
        EEGL_eval = np.vstack((EEGL_eval,pickle.load(open(DIR+'data/ev_EEGL_'+str(i)+'.pkl', "rb"))['L']))
        EEGR_eval = np.vstack((EEGR_eval,pickle.load(open(DIR+'data/ev_EEGR_'+str(i)+'.pkl', "rb"))['R']))

    # Load unlabeled eval data and format it
    EEG_eval = pickle.load(open(DIR+'data/ev_EEG_1.pkl', "rb"))['E'] # (9, 11)
    for i in range(2,10):
        EEG_eval = np.vstack((EEG_eval,pickle.load(open(DIR+'data/ev_EEG_'+str(i)+'.pkl', "rb"))['E']))

    # Load channel locations
    loc = loadmat('./BCICIV2a_loc.mat')['loc']

    return DIR, loc, EEG_eval, EEGL_eval, EEGR_eval, EEGL_train, EEGR_train

def CSP_multiclass(input_data, num_filt): # not working
    # based on: https://github.com/spolsley/common-spatial-patterns/blob/master/CSP.py 
    # input_data : (classes, trials, chans, samples)
    # num_filt : number of spatial filters
    # -> (classes * filters, chans)

    filters = ()
    n_classes = len(input_data)
    # For each class c, find the mean variances Rc and not_Rc, which will be used to compute spatial filter SFc
    for c in range(n_classes):
        # Find Rc
        Rc = cov(input_data[c][0])
        n_trials = input_data.shape[1]
        for trial in range(1,n_trials):
            Rc += cov(input_data[c][trial])
        Rc = Rc / n_trials

        # Find not_Rc
        count = 0
        not_Rc = Rc * 0
        for not_c in [task for task in range(n_classes) if task != c]:
            for trial in range(n_trials):
                not_Rc += cov(input_data[not_c][trial])
                count += 1
        not_Rc = not_Rc / count

        # Find the spatial filter SFc
        SFc = spatialFilter(Rc,not_Rc)
        # SFc = select_filts(SFc, num_filt)
        # filters += (SFc.T,)
        filters += (SFc[:num_filt],)

        # Special case: only two classes, no need to compute any more mean variances
        if n_classes == 2:
            # filters += (select_filts(spatialFilter(not_Rc,Rc), num_filt).T,)
            filters += (spatialFilter(not_Rc,Rc)[:num_filt],)
            break
    return np.vstack(np.array(filters))

def CSP_from_matlab(input_data, num_filt): # not working
    # input_data : (2, trials, chans, samples)
    # num_filt : number of spatial filters
    # -> (2 * filters, chans)
    n_classes = len(input_data)
    n_trials = input_data.shape[1]
    n_channels = input_data.shape[2]
    n_samples = input_data.shape[3]
    # for i = 1:n_classes
    #     for j = 1:n_trials(i)
    #         cov_classes{i}{j} = cov(input_classes{i}{j}',1)/trace(cov(input_classes{i}{j}',1));
    #     end
    # end
    cov_data = np.zeros((n_classes, n_trials, n_channels, n_channels))
    for c in range(n_classes):
        for trial in range(n_trials):
            cov_data[c,trial] = np.cov(input_data[c,trial])/np.trace(np.cov(input_data[c,trial]))
    # R = cell(1,n_classes);
    # for i = 1:n_classes
    #     R{i} = zeros(n_channels, n_channels);
    #     for j = 1:n_trials(i)
    #         R{i} = R{i}+cov_classes{i}{j};
    #     end
    #     R{i} = R{i}/n_trials(i);
    # end
    # Rsum = R{1} + R{2};
    R = np.zeros((n_classes, n_channels, n_channels))
    for c in range(n_classes):
        for trial in range(n_trials):
            R[c] += cov_data[c,trial]
        R[c] = R[c]/n_trials
    Rsum = R[0] + R[1]

    for c in range(n_classes):
        d_C = np.diag(Rsum).mean() * n_channels
        R[c] += d_C * np.eye(n_channels)
    Rsum = R[0] + R[1]
    
    rank_Rsum = np.linalg.matrix_rank(Rsum)
    D, V = np.linalg.eig(Rsum)

    if rank_Rsum < n_channels:
        d = np.diag(D)
        d = d[-rank_Rsum:]
        D = np.diag(d) * np.eye(n_channels)
        V = V[:,-rank_Rsum:]

    # print(D.shape)

    W_T = np.sqrt(la.inv(D* np.eye(n_channels))).dot(V.T)

    S = np.zeros((n_classes, n_channels, n_channels))
    for c in range(n_classes):
        S[c] = W_T.dot(R[c]).dot(W_T.T)

    D, B = np.linalg.eig(S[0])
    ind = np.argsort(D)[::-1] #[::-1] for descending order
    B = B[:,ind]

    result = B.T.dot(W_T)
    dimm = n_classes * num_filt
    m = result.shape[0]
    n = result.shape[1]
    csp_coeff = np.zeros((dimm,n))

    i=0
    for d in range(dimm):
        if d % 2 == 0:
            csp_coeff[d,:] = result[m-i-1,:]
            i+=1
        else:
            csp_coeff[d,:] = result[i,:]
    return csp_coeff



def cov(A):
    # cov takes a matrix A and returns the covariance matrix, scaled by the variance
	Ca = np.dot(A,A.T) / np.trace(np.dot(A,A.T))
	return Ca
    
# spatialFilter returns the spatial filter SFa for mean covariance matrices Ra and Rb
def spatialFilter(Ra,Rb):
	R = Ra + Rb
	D,V = la.eigh(R)

	# CSP requires the eigenvalues D and eigenvector V be sorted in descending order
	ord = np.argsort(D)
	ord = ord[::-1] # argsort gives ascending order, flip to get descending
	D = D[ord]
	V = V[:,ord]

	# Find the whitening transformation matrix
	P = np.dot(np.sqrt(la.inv(np.diag(D))), V.T)

	# The mean covariance matrices may now be transformed
	Sa = np.dot(P, np.dot(Ra,P.T))
	Sb = np.dot(P, np.dot(Rb,P.T))

	# Find and sort the generalized eigenvalues and eigenvector
	D1,V1 = la.eig(Sa,Sb)
	ord1 = np.argsort(D1)
	ord1 = ord1[::-1]
	D1 = D1[ord1]
	V1 = V1[:,ord1]

	# The projection matrix (the spatial filter) may now be obtained
	SFa = np.dot(V1.T, P)
	return SFa.astype(np.float32)

# def select_filts(filt, col_num):
#     columns = np.arange(0,col_num)
#     f = filt[:, columns]
#     for ij in range(col_num):
#         f[:, ij] = f[:, ij]/np.linalg.norm(f[:, ij])
#     return f

def apply_CSP_filter(input_data, filter):
    # input_data : (classes, trials, chans, samples)
    # filter : (classes * filters, chans)
    # -> (classes, trials, classes * filters, samples)

    num_classes = input_data.shape[0]
    num_trials = input_data.shape[1]

    output_data = ()
    for c in range(num_classes):
        class_data = ()
        for trial in range(num_trials):
            class_data += (filter.dot(input_data[c,trial,:,:]),)
        output_data += (np.array(class_data),)
    output_data = np.array(output_data)

    return output_data

def log_norm_band_power(input_data):
    # input_data : (classes, trials, classes * filters, samples)
    # -> (classes, trials, classes * filters)

    num_classes = input_data.shape[0]
    num_trials = input_data.shape[1]

    output_data = ()
    for c in range(num_classes):
        class_data = ()
        for trial in range(num_trials):
            power = np.var(input_data[c][trial], axis=1)
            norm_power = power/power.sum()
            class_data += (np.log(norm_power),)
        output_data += (np.array(class_data),)
    output_data = np.array(output_data)

    return output_data


######################################################################

def train(train_data_1, train_data_2, numFilt):

    numTrials_1 = np.size(train_data_1,0)
    numTrials_2 = np.size(train_data_1,0)

    # train the CCACSP filters 
    ccacsp_filts = calc_CCACSP(train_data_1, train_data_2, numFilt)

    # extract the features
    train_filt_1 = apply_CCACSP(train_data_1, ccacsp_filts, numFilt)
    train_logP_1  = np.squeeze(np.log(np.var(train_filt_1, axis=2)))

    train_filt_2 = apply_CCACSP(train_data_2, ccacsp_filts, numFilt)
    train_logP_2  = np.squeeze(np.log(np.var(train_filt_2, axis=2)))

    # define the classifier
    clf = lda.LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    X = np.concatenate((train_logP_1, train_logP_2), axis=0)

    y1 = np.zeros(numTrials_1)
    y2 = np.ones(numTrials_2)
    y = np.concatenate((y1, y2))

    # train the classifier 
    clf.fit(X, y)

    return ccacsp_filts, clf


def test(test_data, ccacsp_filts, clf):

    total_filts = np.size(ccacsp_filts,1)

    # test the classifier on the test data
    test_filt = np.matmul(ccacsp_filts.transpose(), test_data)
    # print(test_filt.shape)
    test_logP  = np.squeeze(np.log(np.var(test_filt, axis=2)))
    # print(test_logP.shape)
    # test_logP = np.reshape(test_logP,(1,total_filts))

    return clf.predict_proba(test_logP)[:,0]


def calc_CCACSP(x1,x2, numFilt):
    
    
    num_trials_1 = np.size(x1,0) 
    num_trials_2 = np.size(x2,0) 

    # number of channels and time samples should be the same between x1 and x2
    n_samps = np.size(x1,2)
    n_chans = np.size(x1,1) 

    c1_shifted = np.zeros([n_chans,n_chans])
    c2_shifted = np.zeros([n_chans,n_chans])
    c1 = np.zeros([n_chans,n_chans])
    c2 = np.zeros([n_chans,n_chans])

    range0 = range(0,n_samps-2)
    range1 = range(1,n_samps-1)
    range2 = range(2,n_samps)

    # estimate the covariances 
    for ik in range(num_trials_1):
        Samp = x1[ik]
        temp1 = 0.5*(Samp[:,range0]+Samp[:,range2])
        temp2 = Samp[:,range1]
        c1_shifted = c1_shifted+my_cov(temp2, temp1)/np.trace(my_cov(temp2, temp1))

        c1 = c1+np.cov(x1[ik])/np.trace(np.cov(x1[ik]))

    c1_shifted = np.divide(c1_shifted,num_trials_1)
    c1 = np.divide(c1,num_trials_1)

    for ik in range(num_trials_2):
        Samp = x2[ik]
        temp1 = 0.5*(Samp[:,range0]+Samp[:,range2])
        temp2 = Samp[:,range1]
        c2_shifted = c2_shifted+my_cov(temp2, temp1)/np.trace(my_cov(temp2, temp1))

        c2 = c2+np.cov(x2[ik])/np.trace(np.cov(x2[ik]))

    c2_shifted = np.divide(c2_shifted,num_trials_2)
    c2 = np.divide(c2,num_trials_2)
        

    # taking care of rank deficiency for a more robust result 
    D, V = sp.linalg.eigh(c1+c2) 
    indx = np.argsort(D)
    indx = indx[::-1]
    d = D[indx[0:np.linalg.matrix_rank(c1+c2)]]
    W = V[:,indx[0:np.linalg.matrix_rank(c1+c2)]]
    W_T = np.matmul(np.sqrt(sp.linalg.pinv(np.diag(d))),W.transpose())

    S1 = np.matmul(np.matmul(W_T,c1),W_T.transpose())
    S2 = np.matmul(np.matmul(W_T,c2),W_T.transpose())
    S1_shifted = np.matmul(np.matmul(W_T,c1_shifted),W_T.transpose())
    S2_shifted = np.matmul(np.matmul(W_T,c2_shifted),W_T.transpose())

    # find filters for class 1
    d,v = sp.linalg.eigh(S1_shifted,S1+S2)
    indx = np.argsort(d)
    indx = indx[::-1]
    filts_1 = v.take(indx, axis=1)
    filts_1 = np.matmul(filts_1.transpose(),W_T)
    filts_1 = filts_1.transpose()
    filts_1 = select_filts(filts_1, numFilt)

    # find filters for class 2
    d,v = sp.linalg.eigh(S2_shifted,S1+S2)
    indx = np.argsort(d)
    indx = indx[::-1]
    filts_2 = v.take(indx, axis=1)
    filts_2 = np.matmul(filts_2.transpose(),W_T)
    filts_2 = filts_2.transpose()
    filts_2 = select_filts(filts_2, numFilt)

    # concatenate filters for classes 1 and 2 and return 
    return np.concatenate((filts_1, filts_2), axis=1)

def select_filts(filt, col_num):

    temp = np.shape(filt)
    columns = np.arange(0,col_num)
    #print(columns)
    f = filt[:, columns]
    for ij in range(col_num):
        f[:, ij] = f[:, ij]/np.linalg.norm(f[:, ij])

    return f


def apply_CCACSP(X, f, col_num):

    f = np.transpose(f)

    temp = np.shape(X)
    num_trials = temp[0]

    #dat = np.zeros(np.shape(X), dtype = object)
    dat = np.zeros((num_trials, 2*col_num, temp[2]))
    for ik in range(num_trials):
        dat[ik,:,:] = np.matmul(f,X[ik,:,:])

    return dat

def my_cov(X, Y):
	avg_X = np.mean(X, axis=1)
	avg_Y = np.mean(X, axis=1)

	X = X - avg_X[:,None]
	Y = Y - avg_Y[:,None]

	return np.matmul(X, Y.transpose())