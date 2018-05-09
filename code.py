import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("Agg")
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,RationalQuadratic as RQ, ConstantKernel as C, ExpSineSquared
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC, NuSVC
from sklearn.decomposition import PCA, RandomizedPCA, KernelPCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression as LR,LassoLarsIC
from sklearn.metrics import accuracy_score
import pandas as pd


font = {"family": "serif", "color":  "darkgreen", "weight": "normal", "size": 10}
fonttext = {"family": "serif", "color":  "darkblue", "weight": "normal", "size": 12}
fontT = {"family": "serif", "color":  "darkred", "weight": "normal", "size": 12}
fontL = {"family": "serif", "weight": "normal", "size": 10}
fonttitle = {"family": "serif", "color":  "black", "weight": "normal", "size": 15}
fontsuptitle = {"family": "serif", "color":  "black", "weight": "bold", "size": 20}


## Problem 1
def __gmm(M):
    clf = mixture.GaussianMixture(n_components=2)
    clf.fit(np.array(M).reshape(len(M),1))
    print "Mean, var, P - ",clf.means_[:,0], clf.covariances_[:,0,0], clf.weights_
    means = np.array(clf.means_[:,0])
    var = np.array(clf.covariances_[:,0,0])
    p = clf.weights_[0]
    return means, var, p


def __mixture(mu1,mu2,s1,s2,p):
    z = np.linspace(0,20,10000)
    fz = (p*(np.exp(-(z-mu1)**2/(2*s1**2))/np.sqrt(2*np.pi*s1**2)))+((1-p)*(np.exp(-(z-mu2)**2/(2*s2**2))/np.sqrt(2*np.pi*s2**2)))
    return z, fz

def problem1():
    np.random.seed(1)
    mu1,mu2 = 3,15
    s1,s2 = 1,2
    n = 1000
    p = 0.1
    M = []
    Z = np.random.binomial(1,p,size=n)
    for i in range(n):
        if Z[i] == 1: M.append(np.random.normal(mu1,s1,size=1)[0])
        else: M.append(np.random.normal(mu2,s2,size=1)[0])
        pass
    z,fz = __mixture(mu1,mu2,s1,s2,p)
    fig, axes = plt.subplots(figsize=(8,8),nrows=3,ncols=1,dpi=120)
    fig.subplots_adjust(hspace=0.5)
    ax = axes[0]
    ax.hist(M,50,histtype='step',color='b',normed=False)
    ax.set_xlabel(r'$y_i$',fontdict=fontT)
    ax.set_xlim(0,20)
    #ax.set_ylabel(r'$f(x|\mu_1,\mu_2,\sigma_1^2,\sigma_2^2,p)$',fontdict=fontT)
    ax.text(0.3,0.8,r"$Histogram$",horizontalalignment="center", verticalalignment="center",
                                                    transform=ax.transAxes,fontdict=font)
    ax = axes[1]
    ax.set_xlabel(r'$y_i$',fontdict=fontT)
    ax.set_ylabel(r'$f(x|\mu_1,\mu_2,\sigma_1^2,\sigma_2^2,p)$',fontdict=fontT)
    ax.plot(z,fz,'k')
    ax.set_xlim(0,20)
    ax.text(0.3,0.8,r"$Distribution$",horizontalalignment="center", verticalalignment="center",
                                        transform=ax.transAxes,fontdict=font)

    ax = axes[2]
    ax.set_xlabel(r'$y_i$',fontdict=fontT)
    #ax.set_ylabel(r'$f(x|\mu_1,\mu_2,\sigma_1^2,\sigma_2^2,p)$',fontdict=fontT)
    ax.hist(M,50,histtype='step',color='b',normed=True,label=r"$Hist$")
    ax.plot(z,fz,'k',label=r"$Dist$",linewidth=0.75)
    ax.text(0.3,0.8,r"$Normalized$",horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes,fontdict=font)
    ax.set_xlim(0,20)
    mean, var, p = __gmm(M)
    z,fz = __mixture(mean[0],mean[1],var[0]**0.5,var[1]**0.5,p)
    ax.plot(z,fz,'m-.',label=r"$Dist_{EM}$")
    ax.legend(loc=1,prop=fontL)
    fig.savefig('1a.png')
    return



## Problem 2
import statsmodels.formula.api as smf
import statsmodels.api as sm

def __lasso_selected(data,data_test, response):
    X = data.drop([response],axis=1).as_matrix()
    y = np.array(data[response].tolist()).reshape((len(data),1))
    #X = sm.add_constant(X)
    #model = sm.OLS(y,X)
    #m = model.fit_regularized(refit=True)
    #yp = m.predict(data_test)
    reg = LassoLarsIC(criterion='bic')
    print y.shape,X.shape
    reg.fit(X,y)
    x = data_test.drop([response],axis=1).as_matrix().reshape((len(data_test),len(data_test.keys())-1))
    yp = reg.predict(x)
    te = np.mean((yp-np.array(data_test[response].tolist()))**2)
    print reg.coef_,te
    return

def __backward_selected(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            S = set(list(remaining))
            S.remove(candidate)
            formula = "{} ~ {} + 1".format(response,
                    ' + '.join(S))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
            pass
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            pass
        pass
    return list(remaining), selected

def __forward_selected(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                    ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
            pass
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            pass
        pass
    return selected, list(remaining)

def __zeroone(x):
    x = np.array(x)
    X = (x-min(x))/(max(x)-min(x))
    return X

def __smooth(x,window_len=51,window="hanning"):
    x = np.array(x)
    if x.ndim != 1: raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len: raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[d/2:-d/2]
    return y

def __plot_curve(P,data,data_test,response,ax,txt, prime):
    R2_score = []
    AIC = []
    BIC = []
    LLf = []
    TRe = []
    TEe = []
    SL = []
    F = []
    for s in P:
        formula = "{} ~ {} + 1".format(response, ' + '.join(SL + [s]))
        F.append(formula)
        m = smf.ols(formula, data).fit()
        SL.append(s)
        R2_score.append(m.rsquared_adj)
        AIC.append(m.aic)
        BIC.append(m.bic)
        LLf.append(m.llf)
        yp = m.predict(data_test)
        TEe.append(np.mean((yp-np.array(data_test[response].tolist()))**2))
        pass
    print TEe
    ax.plot(range(9),__zeroone(__smooth(R2_score,7)),'r',label=r"$R^2_{adj}$")
    ax.plot(range(9),__zeroone(__smooth(AIC,7)),'b',label=r"$AIC$")
    ax.plot(range(9),__zeroone(__smooth(BIC,7)),'k',label=r"$BIC$")
    #ax.plot(__zeroone(__smooth(LLf,7)),'g',label=r"$R^2_{adj}$")
    ax.legend(loc=1)
    print len(R2_score)
    ax.set_xticklabels([r'$m_0$',r'$m_1$',r'$m_2$',r'$m_3$',r'$m_4$',r'$m_5$',r'$m_6$',r'$m_7$',r'$m_8$',r'$m_9$'])
    if prime: ax.set_xticklabels([r'$m^i_0$',r'$m^i_1$',r'$m^i_2$',r'$m^i_3$',r'$m^i_4$',r'$m^i_5$',r'$m^i_6$',r'$m^i_7$',r'$m^i_8$',r'$m^i_9$'])
    ax.set_ylabel(r'$W_{relative}$',fontdict=fontT)
    ax.set_xlabel(r'$Parameters$',fontdict=fontT)
    ax.legend(loc="lower right")
    ax.text(0.8,0.8,txt,horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,fontdict=font,bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return

def problem2():
    np.random.seed(1)
    df = pd.read_csv("Data_Stat5526_FinalExam/LAozone.txt")
    df['ozone'] = np.array(df['ozone'].tolist())**(1./3.)
    X = df
    X_train, X_test = train_test_split(X, test_size=1.0/3.0, random_state=42)
    
    # Forward Selection
    S,R = __forward_selected(X_train, 'ozone')
    response = 'ozone'
    P = S+R
    SL = []
    for s in P:
        print "{} ~ {} + 1".format(response, ' + '.join(SL + [s]))
        SL.append(s)
    fig = plt.figure(figsize=(5,10),dpi=120)
    ax = fig.add_subplot(211)
    __plot_curve(S+R, X_train, X_test,'ozone',ax,r'(a) $Forward(OLS)$',False)
    S,R = __backward_selected(X_train, 'ozone')
    P = S+R
    SL = []
    for s in P:
        print "{} ~ {} + 1".format(response, ' + '.join(SL + [s]))
        SL.append(s)
    ax = fig.add_subplot(212)
    __plot_curve(S+R, X_train, X_test,'ozone',ax,r'(a) $Backward(OLS)$',True)
    #fig.savefig('2a.png')
    __lasso_selected(X_train,X_test,'ozone')
    return


#problem2()


## Problem 3

def __lda(X,y,x,solver):
    lda = LDA(solver=solver, store_covariance=True)
    lda.fit(X,y)
    y_p = lda.predict(x)
    score = lda.predict_proba(x)
    return y_p,score

def __qda(X,y,x):
    qda = QDA(store_covariance=True)
    qda.fit(X,y)
    y_p = qda.predict(x)
    score = qda.predict_proba(x)
    return y_p,score

def __svm(X,y,x,solver):
    if solver=='svc': svm = SVC(kernel='rbf',probability=True)
    elif solver=='nu': svm = NuSVC(kernel='rbf',probability=True)
    svm.fit(X,y)
    y_p = svm.predict(x)
    score = svm.predict_proba(x)
    return y_p, score

def __conf_matrix(y_pred,y):
    matrix = np.zeros((3,3))
    h,c,a = 'Hispanic', 'Caucasian', 'African American'
    for I,p in enumerate(y_pred):
        if p==y[I] and p==h: matrix[0,0] += 1
        elif p==y[I] and p==c: matrix[1,1] += 1
        elif p==y[I] and p==a: matrix[2,2] += 1
        elif p!=y[I] and p==h and y[I]==c: matrix[1,0] += 1
        elif p!=y[I] and p==h and y[I]==a: matrix[2,0] += 1
        elif p!=y[I] and p==c and y[I]==h: matrix[0,1] += 1
        elif p!=y[I] and p==c and y[I]==a: matrix[2,1] += 1
        elif p!=y[I] and p==a and y[I]==h: matrix[0,2] += 1
        elif p!=y[I] and p==a and y[I]==c: matrix[1,2] += 1
        pass
    return matrix

def __plot_roc(y_test,y_score,ax,txt):
    print y_test.shape,y_score.shape
    Y = label_binarize(y_test, classes=['Hispanic', 'Caucasian', 'African American'])
    n_classes = Y.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        pass
    fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    C = ['r','k','b']
    CC = ['Hisp', 'Cauc', 'AfAm']
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], color=C[i], label=r'$ROC [%s](\bar{A} = %0.2f)$' % (CC[i],roc_auc[i]))
        pass
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel(r'$FPR$',fontdict=fontT)
    ax.set_ylabel(r'$TPR$',fontdict=fontT)
    ax.legend(loc="lower right")
    ax.text(0.8,0.8,txt,horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,fontdict=font,bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return

def __pca(X,solver,C=None,k = 'linear'):
    if solver=='pca': 
        pca = PCA(n_components=C)
        pca_result = pca.fit_transform(X)
        variance_ratio = pca.explained_variance_ratio_
        return pca_result, variance_ratio
    elif solver=='kpca': 
        pca = KernelPCA(n_components=C,kernel=k,gamma=1./3.,degree=2,alpha=0.5,eigen_solver='dense')
        pca_result = pca.fit_transform(X)
        variance = pca.lambdas_ 
        variance_ratio = variance/sum(variance)
        return pca_result, variance_ratio
    elif solver=='rpca': 
        pca = RandomizedPCA(n_components=C,whiten=True)
        pca_result = pca.fit_transform(X)
        variance_ratio = pca.explained_variance_ratio_
        return pca_result, variance_ratio
    else: return None

def __drop(a,c,r):
    a = np.delete(a,r,0)
    a = np.delete(a,c,1)
    return a

def __mis_err(cm):
    T = np.sum(cm)
    TP2 = cm[2,2]
    TN2 = np.sum(__drop(cm,2,2))
    m2 = 1.-((TP2+TN2)/T)
    TP1 = cm[1,1]
    TN1 = np.sum(__drop(cm,1,1))
    m1 = 1.-((TP1+TN1)/T)
    TP0 = cm[0,0]
    TN0 = np.sum(__drop(cm,0,0))
    m0 = 1.-((TP0+TN0)/T)
    return [m0, m1, m2]

def __plot_pca(r,vr,ax,txt,c='navy'):
    ax.plot(np.cumsum(vr), color=c, linestyle='--')
    ax.axvline(3,color='k',linestyle="-.")
    ax.set_xlim([0, 16])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel(r'$Parameters$',fontdict=fontT)
    ax.set_ylabel(r'$Var_{ratio}(n=3)=%.2f$'%sum(vr[:3]),fontdict=fontT)
    ax.text(0.8,0.8,txt,horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,fontdict=font,bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return

def problem3():
    X_names = ['D8S1179','D21S11','D7S820','CSF1PO','D3S1358','TH01','D13S317','D16S539','D2S1338','D19S433','vWA','TPOX','D18S51','D5S818','FGA']
    Y_names = ['population']
    np.random.seed(1)
    df_train = pd.read_csv("Data_Stat5526_FinalExam/geno_train.txt")
    df_test = pd.read_csv("Data_Stat5526_FinalExam/geno_test.txt")
    X_train,X_test = df_train[X_names].as_matrix(),df_test[X_names].as_matrix()
    y_train,y_test = df_train[Y_names].as_matrix()[:,0],df_test[Y_names].as_matrix()[:,0]
    print df_train.population.unique(), df_test.population.unique()
    keys = df_train.population.unique()

    ## LDA
    y_svd,s_svd = __lda(X_train,y_train,X_test,"svd")
    y_lsqr,s_lsqr = __lda(X_train,y_train,X_test,"lsqr")
    y_eigen,s_eigen = __lda(X_train,y_train,X_test,"eigen")
    M_svd = __conf_matrix(y_svd,y_test)
    M_lsqr = __conf_matrix(y_lsqr,y_test)
    M_eigen = __conf_matrix(y_eigen,y_test)
    print __mis_err(M_svd)
    print __mis_err(M_lsqr)
    print __mis_err(M_eigen)
    #print M_svd
    #print "Confusion matrix for LDA - \n",M_svd, "\n",M_lsqr,"\n", M_eigen
    fig = plt.figure(figsize=(10,10),dpi=120)
    ax = fig.add_subplot(321)
    __plot_roc(y_test,s_svd,ax,r'(a) $LDA\left(Solver=SVD\right)$')
    ax = fig.add_subplot(322)
    __plot_roc(y_test,s_lsqr,ax,r'(b) $LDA\left(Solver=LSQR\right)$')
    ax = fig.add_subplot(323)
    __plot_roc(y_test,s_eigen,ax,r'(c) $LDA\left(Solver=Eigen\right)$')
    
    ## QDA
    y_pred,s = __qda(X_train,y_train,X_test)
    M = __conf_matrix(y_pred,y_test)
    print __mis_err(M)
    #print "Confusion matrix for QDA - \n",M
    ax = fig.add_subplot(324)
    __plot_roc(y_test,s,ax,r'(d) $QDA$')

    ## SVM
    y_svc,s_svc = __svm(X_train,y_train,X_test,'svc')
    y_nu,s_nu = __svm(X_train,y_train,X_test,'nu')
    M_svc = __conf_matrix(y_pred,y_test)
    M_nu = __conf_matrix(y_nu,y_test)
    print __mis_err(M_svc)
    print __mis_err(M_nu)
    #print "Confusion matrix for SVM - \n",M_svc,"\n",M_nu
    ax = fig.add_subplot(325)
    __plot_roc(y_svc,s_svc,ax,r'(e) $SVM\left(Solver=SVC\right)$')
    ax = fig.add_subplot(326)
    __plot_roc(y_nu,s_nu,ax,r'(e) $SVM\left(Solver=SVC\right)$')

    #fig.savefig('3a.png')


    #####
    # PCA Decomposition
    #####
    pca_r, pca_vr = __pca(X_train,'pca')
    rca_r, rca_vr = __pca(X_train,'rpca')
    kca_r, kca_vr = __pca(X_train,'kpca')
    kca_r0, kca_vr0 = __pca(X_train,'kpca','poly')
    kca_r1, kca_vr1 = __pca(X_train,'kpca','rbf')
    fig = plt.figure(figsize=(5,10),dpi=120)
    ax = fig.add_subplot(311)
    __plot_pca(pca_r,pca_vr,ax,r'(a) $PCA$')
    ax = fig.add_subplot(312)
    __plot_pca(rca_r,rca_vr,ax,r'(b) $RandomPCA$')
    ax = fig.add_subplot(313)
    #__plot_pca(kca_r0,kca_vr0,ax,r'(c) $KernelPCA(K)$','r')
    #__plot_pca(kca_r1,kca_vr1,ax,r'(c) $KernelPCA(K)$','m')
    __plot_pca(kca_r,kca_vr,ax,r'(c) $KernelPCA(K)$')
    #fig.savefig("3b.png")
    
    ###
    # LDA, QDA, SVM with 3 altered eigens
    ###

    X_train_tx, vr = __pca(X_train,'pca',C=3)
    X_test_tx, vr = __pca(X_test,'pca',C=3)
    
    ## LDA
    y_svd,s_svd = __lda(X_train_tx,y_train,X_test_tx,"svd")
    y_lsqr,s_lsqr = __lda(X_train_tx,y_train,X_test_tx,"lsqr")
    y_eigen,s_eigen = __lda(X_train_tx,y_train,X_test_tx,"eigen")
    M_svd = __conf_matrix(y_svd,y_test)
    M_lsqr = __conf_matrix(y_lsqr,y_test)
    M_eigen = __conf_matrix(y_eigen,y_test)
    print __mis_err(M_svd)
    print __mis_err(M_lsqr)
    print __mis_err(M_eigen)
    #print "Confusion matrix for LDA - \n",M_svd, "\n",M_lsqr,"\n", M_eigen
    fig = plt.figure(figsize=(10,10),dpi=120)
    ax = fig.add_subplot(321)
    __plot_roc(y_test,s_svd,ax,r'(a) $LDA\left(Solver=SVD\right)$')
    ax = fig.add_subplot(322)
    __plot_roc(y_test,s_lsqr,ax,r'(b) $LDA\left(Solver=LSQR\right)$')
    ax = fig.add_subplot(323)
    __plot_roc(y_test,s_eigen,ax,r'(c) $LDA\left(Solver=Eigen\right)$')
    
    ## QDA
    y_pred,s = __qda(X_train_tx,y_train,X_test_tx)
    M = __conf_matrix(y_pred,y_test)
    print __mis_err(M)
    #print "Confusion matrix for QDA - \n",M
    ax = fig.add_subplot(324)
    __plot_roc(y_test,s,ax,r'(d) $QDA$')

    ## SVM
    y_svc,s_svc = __svm(X_train_tx,y_train,X_test_tx,'svc')
    y_nu,s_nu = __svm(X_train_tx,y_train,X_test_tx,'nu')
    M_svc = __conf_matrix(y_svc,y_test)
    M_nu = __conf_matrix(y_nu,y_test)
    print __mis_err(M_svc)
    print __mis_err(M_nu)
    #print "Confusion matrix for SVM - \n",M_svc,"\n",M_nu
    ax = fig.add_subplot(325)
    __plot_roc(y_svc,s_svc,ax,r'(e) $SVM\left(Solver=SVC\right)$')
    ax = fig.add_subplot(326)
    __plot_roc(y_nu,s_nu,ax,r'(e) $SVM\left(Solver=SVC\right)$')

    #fig.savefig('3c.png')
 
    return

#problem3()


## Problem 4

def __kernel_ridge(fn,X,y,x):
    param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
            "kernel": [ExpSineSquared(l, p)
                for l in np.logspace(-2, 2, 10)
                for p in np.logspace(0, 2, 10)]}
    kr = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
    kr.fit(X, y)
    y_kr = kr.predict(x)
    return y_kr

def __kernel_regress(fn, X, y, x):
    if fn == 0: kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
    elif fn == 1: kernel = C(1.0, (1e-3, 1e3)) * RQ(length_scale=5.0, alpha=5.0, length_scale_bounds=(1e-02, 1e2), alpha_bounds=(1e-02, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(X, y)
    y_pred, sigma = gp.predict(x, return_std=True)
    return y_pred[:,0], sigma

def __plot_2DSurface(x,y,z,ax,c,txt):
    XT = x.reshape((51,51))
    YT = y.reshape((51,51))
    Z = z.reshape((51,51))
    CS = ax.plot_surface(XT,YT,Z,color=c)
    ax.set_xlabel(r'$X_1$',fontdict=fontT)
    ax.set_ylabel(r'$X_2$',fontdict=fontT)
    ax.set_zlabel(r'$Y$',fontdict=fontT)
    ax.text(0.8,0.8,10,txt,
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,fontdict=fonttext,bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return

def __plot_2DContour(x,y,z,ax,txt):
    XT = x.reshape((51,51))
    YT = y.reshape((51,51))
    Z = z.reshape((51,51))
    CS = ax.contour(XT,YT,Z)
    ax.clabel(CS, inline=1, fontdict=fontT)
    ax.set_xlabel(r'$X_1$',fontdict=fontT)
    ax.set_ylabel(r'$X_2$',fontdict=fontT)
    ax.text(0.8,0.8,txt,
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,fontdict=fonttext,bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return

def __plot_line(x,z,z_pred,s,ax,txt,T=False,kr=False):
    XT = x.reshape((51,51))
    Z = z.reshape((51,51))
    Zp = z_pred.reshape((51,51))
    S = s.reshape((51,51))
    if T:
        XT = XT.T
        Z = Z.T
        Zp = Zp.T
        S = S.T
    ax.plot(XT[:,0],Z[:,0],'r.', markersize=2, label="Observations")
    ax.plot(XT[:,0],Zp[:,0],'k',label="Predicted")
    x = XT[:,0]
    z_pred = Zp[:,0]
    sigma = S[:,0]
    if not kr:
        ax.fill(np.concatenate([x, x[::-1]]), np.concatenate([z_pred - 1.9600 * sigma, (z_pred + 1.9600 * sigma)[::-1]]),
                alpha=.5, fc='b', ec='None', label='95% confidence interval')
    ax.legend(loc=0)
    ax.set_xlabel(r'$X_1$',fontdict=fontT)
    if T: ax.set_xlabel(r'$X_2$',fontdict=fontT)
    ax.text(0.2,0.2,txt,
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,fontdict=fonttext,bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_ylabel(r'$Y$',fontdict=fontT)
    ax.set_xlim(0,1)
    return

def problem4():
    X_names = ['X1','X2']
    Y_names = ['Y']
    np.random.seed(1)
    df_train = pd.read_csv("Data_Stat5526_FinalExam/Pr6_training.txt",sep=" ")
    df_test = pd.read_csv("Data_Stat5526_FinalExam/Pr6_test.txt",sep=" ")
    X_train,X_test = df_train[X_names].as_matrix(),df_test[X_names].as_matrix()
    y_train,y_test = df_train[Y_names].as_matrix(),df_test[Y_names].as_matrix()

    ## GPR
    y_rbf,s_rbf = __kernel_regress(0,X_train,y_train,X_test)
    y_rq,s_rq = __kernel_regress(1,X_train,y_train,X_test)

    fig = plt.figure(figsize=(10,10),dpi=120)
    ax = fig.add_subplot(221)
    __plot_line(X_test[:,0],y_test[:,0],y_rbf,s_rbf,ax,r"(a) ")
    ax = fig.add_subplot(222)
    __plot_line(X_test[:,1],y_test[:,0],y_rbf,s_rbf,ax,r"(b) ",True)
    ax = fig.add_subplot(223)
    __plot_line(X_test[:,0],y_test[:,0],y_rq,s_rq,ax,r"(c) ")
    ax = fig.add_subplot(224)
    __plot_line(X_test[:,1],y_test[:,0],y_rq,s_rq,ax,r"(d) ",True)
    fig.text(0.06,0.72,r"$\kappa (x,x')=\sigma^2.exp\left(-\frac{||x-x'||^2}{2\lambda^2}\right)$",
                            horizontalalignment="center", verticalalignment="center",fontdict=fonttitle,rotation=90)
    fig.text(0.06,0.3,r"$\kappa (x,x')=\sigma^2.\left(1+\frac{||x-x'||^2}{2\alpha\lambda^2}\right)^{-\alpha}$",
                        horizontalalignment="center", verticalalignment="center",fontdict=fonttitle,rotation=90)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    #fig.subplots_adjust(hspace=0.2,wspace=0.4)
    plt.show()
    fig.savefig('4a.png')
    plt.close()

    fig = plt.figure(figsize=(10,10),dpi=120)
    ax = fig.add_subplot(241)
    __plot_2DContour(X_test[:,0],X_test[:,1],y_test[:,0],ax,r"(a) $Observations$")
    ax = fig.add_subplot(242)
    __plot_2DContour(X_test[:,0],X_test[:,1],y_rbf,ax,r"(b) $Pred_{mean}$")
    ax = fig.add_subplot(243)
    __plot_2DContour(X_test[:,0],X_test[:,1],y_rbf+s_rbf,ax,r"(c) $C_B=97.5\%$")
    ax = fig.add_subplot(244)
    __plot_2DContour(X_test[:,0],X_test[:,1],y_rbf-s_rbf,ax,r"(d) $C_B=2.5\%$")
    fig.text(0.06,0.72,r"$\kappa (x,x')=\sigma^2.exp\left(-\frac{||x-x'||^2}{2\lambda^2}\right)$",
                horizontalalignment="center", verticalalignment="center",fontdict=fonttitle,rotation=90)

    ax = fig.add_subplot(245)
    __plot_2DContour(X_test[:,0],X_test[:,1],y_test[:,0],ax,r"(e) $Observations$")
    ax = fig.add_subplot(246)
    __plot_2DContour(X_test[:,0],X_test[:,1],y_rq,ax,r"(f) $Pred_{mean}$")
    ax = fig.add_subplot(247)
    __plot_2DContour(X_test[:,0],X_test[:,1],y_rq+s_rq,ax,r"(g) $C_B=97.5\%$")
    ax = fig.add_subplot(248)
    __plot_2DContour(X_test[:,0],X_test[:,1],y_rq-s_rq,ax,r"(h) $C_B=2.5\%$")
    fig.text(0.06,0.3,r"$\kappa (x,x')=\sigma^2.\left(1+\frac{||x-x'||^2}{2\alpha\lambda^2}\right)^{-\alpha}$",
            horizontalalignment="center", verticalalignment="center",fontdict=fonttitle,rotation=90)
    fig.subplots_adjust(hspace=0.2,wspace=0.4)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()
    fig.savefig('4b.png')

    fig = plt.figure(figsize=(10,10),dpi=120)
    ax = fig.add_subplot(221, projection='3d')
    __plot_2DSurface(X_test[:,0],X_test[:,1],y_test[:,0],ax,'b',r"(a) $Observations$")
    ax = fig.add_subplot(222, projection='3d')
    __plot_2DSurface(X_test[:,0],X_test[:,1],y_test[:,0],ax,'lightblue',"")
    __plot_2DSurface(X_test[:,0],X_test[:,1],y_rbf,ax,'k',"")
    __plot_2DSurface(X_test[:,0],X_test[:,1],y_rbf+s_rbf,ax,'lightgreen',"")
    __plot_2DSurface(X_test[:,0],X_test[:,1],y_rbf-s_rbf,ax,'red',r"(b) Mean with CI")
    fig.text(0.06,0.72,r"$\kappa (x,x')=\sigma^2.exp\left(-\frac{||x-x'||^2}{2\lambda^2}\right)$",
                            horizontalalignment="center", verticalalignment="center",fontdict=fonttitle,rotation=90)
    ax = fig.add_subplot(223, projection='3d')
    __plot_2DSurface(X_test[:,0],X_test[:,1],y_test[:,0],ax,'b',r"(c) $Observations$")
    ax = fig.add_subplot(224, projection='3d')
    __plot_2DSurface(X_test[:,0],X_test[:,1],y_rq,ax,'k',"")
    __plot_2DSurface(X_test[:,0],X_test[:,1],y_test[:,0],ax,'lightblue',"")
    __plot_2DSurface(X_test[:,0],X_test[:,1],y_rq+s_rq,ax,'lightgreen',"")
    __plot_2DSurface(X_test[:,0],X_test[:,1],y_rq-s_rq,ax,'red',r"(d) Mean with CI")
    fig.text(0.08,0.3,r"$\kappa (x,x')=\sigma^2.\left(1+\frac{||x-x'||^2}{2\alpha\lambda^2}\right)^{-\alpha}$",
            horizontalalignment="center", verticalalignment="center",fontdict=fonttitle,rotation=90)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()
    fig.savefig('4c.png')


    ## Kernel Ridge Regression
    y_rbf = __kernel_ridge(0,X_train,y_train,X_test)

    fig = plt.figure(figsize=(5,10),dpi=120)
    ax = fig.add_subplot(211)
    __plot_line(X_test[:,0],y_test[:,0],y_rbf,0*s_rbf,ax,r"(a) ",False,True)
    ax = fig.add_subplot(212)
    __plot_line(X_test[:,1],y_test[:,0],y_rbf,0*s_rbf,ax,r"(b) ",True,True)
    fig.text(0.5,0.9,r"$\kappa (x,x')=\sigma^2.exp\left(-\frac{2\sin^2\left(\pi.|x-x'|/p\right)}{\lambda^2}\right)$",
            horizontalalignment="center", verticalalignment="center",fontdict=fonttitle)
    plt.show()
    #fig.savefig('4d.png')
    plt.close()

    fig = plt.figure(figsize=(5,10),dpi=120)
    ax = fig.add_subplot(211)
    __plot_2DContour(X_test[:,0],X_test[:,1],y_test[:,0],ax,r"(a) $Observation$")
    ax = fig.add_subplot(212)
    __plot_2DContour(X_test[:,0],X_test[:,1],y_rbf,ax,r"(b) $Pred$")
    mse = np.mean((y_test[:,0] - y_rbf)**2)
    fig.text(0.5,0.93,r"$\kappa (x,x')=\sigma^2.exp\left(-\frac{2\sin^2\left(\pi.|x-x'|/p\right)}{\lambda^2}\right)$"+"\nMSE="+str(np.round(mse,3)),
                        horizontalalignment="center", verticalalignment="center",fontdict=fonttitle)
    plt.show()
    #fig.savefig('4e.png')
    return

run = False

#problem1()
problem3()
#problem2()
#problem4()
if run:
    problem1()
    problem4()
    problem3()
    problem2()
