
from pl_packages import *

#Finds emperical distribution of data
def ecdf_points(data):
    L=len(data)
    data_array=[]
    for i in range(L):
        data_array.append(data[i])
    np.asarray(data_array,float)
    x=np.sort(data_array)
    y=np.arange(0,1,1/L)
    X=[x[0]]
    Y=[y[0]]
    for i in range(0,L-1):
        if(x[i]!=x[i+1]):
            X.append(x[i+1])
            Y.append(y[i+1])
    X=np.asarray(X,float)
    Y=np.asarray(Y,float)
    return X,Y

#Finds tail of data
def tail(data):
    x,y=ecdf_points(data)
    return x, 1-y


#######################################################################

#Finds index of array with value of that index closest to the value imputted
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#Power law function
def power_law(x,a,b):
    return a*x**b

###########################################################################

'''Functions for Clauset et al paper'''

#Get's the MLE power beta parameter
def mle_pl_beta(x,x_min):
    n=len(x)
    return n/np.sum(np.log(x/x_min))

#Fit's the power law with MLE outputting parameters, predictions and index for inputted
#minimum x value for sample fit
def get_power_law_fit(sample,x_m):
    pl_sample=sample[sample>=x_m]
    y_n=len(pl_sample)/len(sample)
    x,y=tail(pl_sample)
    beta=mle_pl_beta(pl_sample,x_m)
    alpha=y_n*x_m**beta
    y_pred=power_law(x,alpha,-beta)
    return x,y_pred,alpha,beta

#This slightly adjusted function fits the unbiased MLE
def get_power_law_fit1(sample,x_m):
    pl_sample=sample[sample>=x_m]
    ns=len(pl_sample)
    y_n=ns/len(sample)
    x,y=tail(pl_sample)
    beta=(ns-1)/ns*mle_pl_beta(pl_sample,x_m)
    alpha=y_n*x_m**beta
    y_pred=power_law(x,alpha,-beta)
    return x,y_pred,alpha,beta

#KS statistic
def KS_stat(y,y_pred):
    return np.max(np.abs(y-y_pred))

#Get's estimate for x_min. Outputs this estimate which is the minimum of the
#KS-statistics also outputted
def x_min_pred(data):
    es1=[]
    x,y=tail(data)
    for i in range(len(x)-2):
        xs,ys,a,b=get_power_law_fit(data,x[i])
        y_norm=y[i]
        yn=y[i:]/y_norm
        ysn=ys/y_norm
        e1=KS_stat(yn,ysn)
        es1.append(e1)
    ind1=np.argmin(es1)
    x_min_pred=x[ind1]
    return es1,x_min_pred

#As above function but chooses x_min only within pre-chosen interval [x_l,x_u]
def x_min_pred_interval(data,x_l,x_u):
    data=np.sort(data)
    if (x_l>=data[0] )& (x_u<data[-2]):
        es1=[]
        x,y=tail(data)
        ind=np.where((x_l<=x) & (x<=x_u))[0]
        for i in ind:
            xs,ys,a,b=get_power_law_fit(data,x[i])
            y_norm=y[i]
            yn=y[i:]/y_norm
            ysn=ys/y_norm
            e1=KS_stat(yn,ysn)
            es1.append(e1)
        ind1=np.argmin(es1)+ind[0]
        x_min_pred=x[ind1]
        return es1,x_min_pred
    else:
        return 'Interval chosen out of range'

#Goodness of fit test for power law
def get_p_val(x_m,b,K,N,n):
    Ks=[]
    for i in range(N):
        sample=pareto.rvs(pareto_fit_vec[3], scale=x_m, size=n)
        x,y=tail(sample)
        vec=get_power_law_fit(sample,x_m)
        Ks.append(KS_stat(y,vec[1]))
    p=len(np.where(Ks>K)[0])/N
    return p

############################################################################

#Linear regression fit - zero intercept
#https://math.stackexchange.com/questions/3297060/linear-regression-without-intercept-formula-for-slope

def pl_reg_fit(sample,x_m):
    pl_sample=sample[sample>=x_m]
    y_n=len(pl_sample)/len(sample)
    x,y=tail(pl_sample)
    x_lin_reg=np.log10(x/x_m)
    y_lin_reg=np.log10(y)
    b=np.sum(x_lin_reg*y_lin_reg)/np.sum(x_lin_reg*x_lin_reg)
    beta=-b
    alpha=y_n*x_m**(beta)
    y_reg_pred=alpha*x**(-beta)
    return [x,y_reg_pred,alpha,beta]

#Expected mean function linear regression
def lr_mean_est(n,g):
    return np.log((np.exp(1)-(np.log(n))**(g)/n))

#Linear regression fit, approximately unbiased
def pl_reg_fit1(sample,x_m,g):
    pl_sample=sample[sample>=x_m]
    n=len(pl_sample)
    y_n=n/len(sample)
    x,y=tail(pl_sample)
    x_lin_reg=np.log10(x/x_m)
    y_lin_reg=np.log10(y)
    b=np.sum(x_lin_reg*y_lin_reg)/np.sum(x_lin_reg*x_lin_reg)
    beta=-b/lr_mean_est(n,g)
    alpha=y_n*x_m**(beta)
    y_reg_pred=alpha*x**(-beta)
    return [x,y_reg_pred,alpha,beta]

##############################################################################

#Fit power law with non-linear regression

#b free param, a = P(X>x_m)*x_m^b
def min_non_lin_pl(b,x_m,x,y):
    return y-power_law(x,x_m**b,-b)

def pl_nlr_fit(sample,b0,x_m):
    pl_sample=sample[sample>=x_m]
    y_n=len(pl_sample)/len(sample)
    x,y=tail(pl_sample)
    res=least_squares(min_non_lin_pl,b0,args=(x_m,x,y))
    b=res.x[0]
    a=y_n*x_m**b
    y_pred=power_law(x,a,-b)
    return x,y_pred,a,b
