import time
import re
import inspect

import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.optimize as opt
import scipy.interpolate as intp
import patsy as ps
import statsmodels.api as sm
import sklearn.metrics as met
import sklearn.linear_model as lm

__version__ = "0.16.3"
__name__ = "MLtools"


def validpar(f, x):
    '''
    Avoid invalid parameters for a function
    
    Parameters
    ----------
    f : function, which cannot accept invalid parameters
    x : dict, including pairs of parameter name and value
    
    Returns
    -------
    op : dict, valid parameters for the function
    '''
    op = dict(i for i in x.items() if i[0] in inspect.signature(f).parameters.keys())
    return(op)


def strnum(x, f_reduce=max):
    '''
    Convert string to number, only keep numerical part (0-9 or .), otherwise return np.nan
    
    Parameters
    ----------
    x : string, to convert
    
    Returns
    -------
    op : float
    '''
    if isinstance(x, str):
        numL = re.findall(r'\d[\.\d]+', x)
        if numL:
            return(f_reduce([float(i) for i in numL]))
        else:
            return(np.nan)
    else:
        return(x)

def plyargs(f, argL, argname, f_con=list, argcon={}, **kwargs):
    '''
    Apply function on a list of arguments
    
    Parameters
    ----------
    f : function, to apply arguments
    argL : list (of lists), iterated argument values
    argname : list, argument names, same length as argL
    f_con : function, to concat iterated outputs of f
    argcon : dict, arguments of f_con
    **kwargs : arguments of f
    
    Returns
    -------
    op : list-like, iterated outputs of f
    '''
    op = f_con(map(lambda arg: f(**dict(zip(*[argname, arg])), **kwargs), zip(*argL)), **argcon)
    return(op)


def collinearvif(df):
    op = pd.Series(np.diag(np.linalg.inv(df.corr())), index=df.columns, name = "CollinearVIF")
    return(op)


def summary(df, n=5, pct=[0.1, 0.5, 0.9]):
    '''
    Summarize pandas Series for data type, sample size, numerical statistics and frequency
    
    Parameters
    ----------
    df : DataFrame, to summarize
    n : int, the number of foremost frequent categories of frequency table
    pct : list, percentiles for numerical Series statistics

    Returns
    -------
    op : DataFrame, summary of Series
    '''
    
    def freq(s):
        op = pd.value_counts(s)
        op = pd.concat([pd.Series(op.index[:n]).rename(lambda x: "FreqCat{}".format(x+1)), pd.Series(op.values[:n]).rename(lambda x: "FreqVal{}".format(x+1)).T, pd.Series(op.iloc[n:].sum(), index=["Freq_Others"])])
        return(op)
    op = pd.concat([df.dtypes.rename("Type"), df.notnull().sum().rename("N"), df.describe(pct).iloc[1:].T, df.apply(freq).T], axis=1).loc[df.columns]
    return(op)


## Clean Data

def CLtimenum(s):
    '''
    Convert datetime variable to numeric variable
    '''
    op = pd.DatetimeIndex(s)
    op = pd.get_dummies(pd.DataFrame(dict(zip(*[["{}__{}".format(s.name, i) for i in ["Month", "Year"]], [op.month, op.year]])), index=s.index).astype("O"))
    return(op)


def CLnormal(df):
    '''
    Normalize the distribution of numeric DataFrame to standard normal distribution by rank
    '''
    op = pd.DataFrame(st.norm.ppf((df.rank(pct = True) - 0.5/df.shape[0]).fillna(0.5)), index=df.index, columns=df.columns, dtype = "float32")
    return(op)


def CLscale(df):
    '''
    Standardize the distribution of numeric DataFrame to mean 0 and standard deviation 1
    '''
    op = ((df - df.mean())/df.std()).fillna(0).astype("float32")
    return(op)


def CLsparse_cat(s, sp=0.01):
    sfreq = pd.value_counts(s) > sp*s.shape[0]
    op = s.where(s.isin(sfreq.index[sfreq]), "others")
    return(op)


def CLsparse(s):
    op = 1 - s.isnull().mean()
    if op > 0:
        op -= pd.value_counts(s).iloc[0]/s.shape[0]
    return(op)


def CLdata(df, sp=0, cor=1, f_norm = CLscale, formula=None, **kwargs):
    '''
    Clean a pandas DataFrame according to dtype (numeric or object)
    
    Parameters
    ----------
    df : DataFrame, data to clean
    sparse : float: (0, 1], ratio threshold of unique and missing values to delete a variable
    f_norm : function, use to standardize numeric variables
    formula : str, add transformed or interaction terms of variables
    cor : float: (0, 1], threshold of absolute correlation to remove later collinear variables
    
    Returns
    -------
    df_clean : DataFrame, cleaned data
    '''
    df_num = df.select_dtypes(["number"])
    df_cat = df.select_dtypes(["object"])
    if(df_cat.shape[1]):
        df_cat = pd.get_dummies(df_cat.apply(CLsparse_cat, sp = sp))
    if(df_num.shape[1] and f_norm):
        df_num = f_norm(df_num)
    df_clean = pd.concat([df_num, df_cat], axis=1)
    if(sp):
        df_clean = df_clean.loc[:, df_clean.apply(CLsparse)>sp]
    if(cor < 1):
        abscor = np.abs(np.tril(np.corrcoef(df_clean, rowvar=0), -1))
        df_clean = df_clean.loc[:, np.all(abscor<cor, axis=1)]
    if(formula):
        df_clean = ps.dmatrix(("0"+"+{}"*df_clean.shape[1]+formula).format(*df_clean.columns), data=df_clean, return_type="dataframe")
    return(df_clean)


## Machine Learning Models


def MDSingleReg(X, Y, X_offset = [], f_model = sm.OLS, par_model = {}, fix_offset = False, **kwargs):
    par = {}
    par.update(par_model)
    if len(X_offset):
        cov = np.hstack([np.ones((len(Y), 1)), X_offset.values])
        if fix_offset:
            model_cov = f_model(Y.values, cov, **par).fit()
            par.update({"offset": np.dot(cov, model_cov.params)})
            cov = np.ones((len(Y), 1))
    else:
        cov = np.ones((len(Y), 1))
    def fit(item):
        model = f_model(Y.values, np.hstack([cov, item[1].values[:, np.newaxis]]), **par).fit()
        return([model.params[-1], model.bse[-1], model.tvalues[-1], model.pvalues[-1]])
    op = pd.DataFrame(list(map(fit, X.iteritems())), index=X.columns, columns=["beta", "std", "t", "p-value"])
    return(op)


def MLsinglereg(xt, yt, xv=None, yv=None, seed=0, par_model={}, ic_offset=[], rank = True, pct = 1, **kwargs):
    '''
    Train a model of statsmodel form
    
    Parameters
    ----------
    xt : DataFrame, training X set
    xv : DataFrame, validation X set
    yt : DataFrame, training Y set
    yv : DataFrame, validation Y set
    seed : int, random seed
    f_model : function, model to train
    par_model : dict, arguments to train f_model
    ic_offset : str, offset column name of X of which the beta is fixed at 1
    
    Returns
    -------
    model : trained model
    '''
    par = {}
    par.update(par_model)
    if len(ic_offset):
        xt_offset = xt[ic_offset]
    else:
        xt_offset = []
    if par.get("family"):
        daw = MDSingleReg(xt.drop(ic_offset, axis = 1), yt, xt_offset, f_model = sm.GLM, par_model = par)
        model = lm.LogisticRegression()
        model.intercept_ = 1/(1/np.mean(yt.values) - 1)
    else:
        daw = MDSingleReg(xt.drop(ic_offset, axis = 1), yt, xt_offset, f_model = sm.OLS, par_model = par)
        model = lm.LinearRegression()
        model.intercept_ = np.mean(yt.values)
    if rank:
        daw["p-value"] = daw["p-value"].rank(pct = True)
    model.coef_ = (daw["beta"].values*(daw["p-value"].values <= pct))[np.newaxis, :]
    return(model)


def MLstatsmodel(xt, yt, xv=None, yv=None, seed=0, f_model=sm.GLM, par_model={"family": sm.families.Binomial}, ic_offset=[], **kwargs):
    '''
    Train a model of statsmodel form
    
    Parameters
    ----------
    xt : DataFrame, training X set
    xv : DataFrame, validation X set
    yt : DataFrame, training Y set
    yv : DataFrame, validation Y set
    seed : int, random seed
    f_model : function, model to train
    par_model : dict, arguments to train f_model
    ic_offset : str, offset column name of X of which the beta is fixed at 1
    
    Returns
    -------
    model : trained model
    '''
    par = {}
    par.update(par_model)
    if len(ic_offset):
        par.update({"offset": xt.loc[:,ic_offset]})
    model = f_model(yt, xt.drop(ic_offset, axis=1).assign(_Int_=1).rename(columns={"_Int_": "(Intercept)"}), **par).fit()
    return(model)


def MLmodel(xt, yt, xv=None, yv=None, seed=0, f_model=lm.LogisticRegression, par_model={}, par_fit={'verbose': False}, **kwargs):
    '''
    Train a model of scikit-learn form
    
    Parameters
    ----------
    xt : DataFrame, training X set
    xv : DataFrame, validation X set
    yt : DataFrame, training Y set
    yv : DataFrame, validation Y set
    seed : int, random seed
    f_model : function, model to train
    par_model : dict, arguments to train f_model
    
    Returns
    -------
    model : trained model
    '''
    par = {"random_state": seed, "n_estimators": max(100, xt.shape[1]), 'learning_rate': 0.05, 'n_jobs': -1, "class_weight": 'balanced',
           "seed": seed, "colsample_bylevel": 1/(1+np.log(xt.shape[1]))}
    par.update(par_model)
    model = f_model(**validpar(f_model, par))
    model.fit(xt, yt.iloc[:,0], **validpar(model.fit, {"eval_set": [(xv, yv.iloc[:,0])], 'verbose': False, **par_fit}))
    return(model)


## Machine Learning Model Functions


def MDtrain(f_model, **kwargs):
    '''
    Call a proper model to train according to f_model
    
    Parameters
    ----------
    f_model : function, model to train
    **kwargs : arguments of function f_model
    
    Returns
    -------
    model : trained model
    '''
    if f_model.__name__ == "GLM":
        model = MLstatsmodel(f_model=f_model, **kwargs)
    elif f_model.__name__ == 'MLsinglereg':
        model = MLsinglereg(**kwargs)
    else:
        model = MLmodel(f_model=f_model, **kwargs)
    return(model)  


def MDpred(model, xv, ic_offset=[], f_loss=met.roc_auc_score, logit=False, **kwargs):
    '''
    Use trained model and validation X to predict Y
    
    Parameters
    ----------
    model : trained model
    xv : DataFrame, validation X to predict Y
    ic_offset : str, offset column name of X for Y
    
    Returns
    -------
    yvp : DataFrame, predicted Y
    '''
    xvo = xv
    if len(ic_offset):
        xv = xv.drop(ic_offset, axis=1)
    if type(model).__name__ == 'GLMResultsWrapper':
        xv = xv.assign(_Int_ = 1).rename(columns={"_Int_": "(Intercept)"})
    if ((type(model).__name__ == 'LogisticRegression')|('Classifier' in type(model).__name__)) & (f_loss.__name__ in ["roc_auc_score"]):
        yvp = model.predict_proba(xv)[:,1]
    elif (type(model).__name__ == 'GLMResultsWrapper') & len(ic_offset):
        yvp = model.predict(xv, offset=xvo.loc[:,ic_offset])
    else:
        yvp = model.predict(xv)
    if (f_loss.__name__ in ["roc_auc_score", "log_loss"]) and logit:
        yvp = np.log(1/(1/(yvp+1e-7) - 1))
    op = pd.DataFrame(yvp, index=xv.index)
    return(op)


def MDweight(model, xt, ic_offset=[], **kwargs):
    '''
    Use trained model and X to get variable weights
    
    Parameters
    ----------
    model : trained model
    xt : DataFrame, get variable name from training X set
    
    Returns
    -------
    op : Series, variable weights of the model
    '''
    if type(model).__name__ in ['Ridge', "Lasso"]:
        op = pd.Series(model.coef_, index=xt.drop(ic_offset, axis=1).columns)
    elif type(model).__name__ in ['LogisticRegression']:
        op = pd.Series(model.coef_[0], index=xt.drop(ic_offset, axis=1).columns)
    elif 'sklearn.ensemble' in type(model).__module__:
        op = pd.Series(model.feature_importances_, index=xt.drop(ic_offset, axis=1).columns)
    elif type(model).__name__ in ["GLMResultsWrapper"]:
        op = pd.concat([model.params, model.pvalues], axis=1, keys=["coef", "P-value"])
    else:
        op = pd.Series()
    return(op)

def Loss(yv, yp, f_loss=met.roc_auc_score, **kwargs):
    '''
    Get loss between true validation Y and predicted Y
    
    Parameters
    ----------
    yv : DataFrame, validation Y
    yp : DataFrame, predicted Y
    f_loss : function, use to calculate loss
    
    Returns
    -------
    op : numeric, loss
    '''
    op = f_loss(np.array(yv).flatten(), np.array(yp).flatten())
    return(op)


def roc(yv, yp, plot=False, **kwargs):
    '''
    Get ROC curve between binary validation Y and predicted Y
    
    Parameters
    ----------
    yv : DataFrame, binary validation Y
    yp : DataFrame, predicted Y
    plot : bool, whether plot a ROC curve
    
    Returns
    -------
    op : DataFrame, elements of an ROC curve including: true positive ratio, false positive ratio, and threshold
    '''
    op = pd.DataFrame(dict(zip(*[["FPR", "TPR", "Threshold"], met.roc_curve(yv, yp)])))
    if plot:
        op.set_index("FPR").plot(ylim=[0, 1], title="AUC: {:.4f}".format(met.auc(op["FPR"], op["TPR"])))
    return(op)


## Model Analysis


def MDweight_analysis(model, xt, **kwargs):
    w = MDweight(model, xt, **kwargs)
    if 'sklearn.ensemble' in type(model).__module__:
        p0 = 1/len(w)
        n_split = sum((i.tree_.feature != -2).sum() for i in np.array(model.estimators_).flatten())
        op = w.to_frame("freq")
        op["std"] = np.sqrt(w*(1-w)/n_split)
        op["Z-score"] = (w - p0)/np.sqrt(p0*(1-p0)/n_split)
        op["p-value"] = st.norm.cdf(-op["Z-score"])
    return(op)


def tree_set(tree_, max_depth = 5):
    def recurse_set(node, node_set_parent, depth):
        name = tree_.feature[node]
        node_set = [{name}]
        for i in node_set_parent:
            node_set_i = i.copy()
            node_set_i.add(name)
            node_set.append(node_set_i)
        if depth < max_depth:
            node_lchild = recurse_set(tree_.children_left[node], node_set, depth+1) if tree_.feature[tree_.children_left[node]] != -2 else []
            node_rchild = recurse_set(tree_.children_right[node], node_set, depth+1) if tree_.feature[tree_.children_right[node]] != -2 else []
            node_set += node_lchild
            node_set += node_rchild
        return(node_set)
    return(recurse_set(0, [], 1))


def MDforest_set(model, xt = None, max_depth = 5, alpha = 0.05):
    asso_S = pd.value_counts(tuple(j) for i in np.array(model.estimators_).flatten() for j in tree_set(i.tree_, max_depth))
    op = []
    ic_name = range(model.n_features_) if xt is None else xt.columns.tolist()
    for i, i_val in asso_S.groupby(asso_S.rename(len).index):
        i_val = i_val.rename(lambda x: ic_name[x[0]]) if i == 1 else pd.Series(i_val.values, [tuple(ic_name[j] for j in x) for x in i_val.index])
        p0 = 1/misc.comb(model.n_features_, i)
        freq0 = i_val.sum()*p0
        if alpha >= 0:
            i_val = i_val[i_val > np.ceil(freq0-st.norm.ppf(alpha*p0)*np.sqrt(freq0*(1-p0)))]
        op.append(i_val)
    return(op)


## Cross-Validation


def Kfolds(x, k=10, seed=1):
    '''
    Use index to create a list of K-folds cross-validation indices
    
    Parameters
    ----------
    x : list, sample index
    k : int, number of folds
    seed : int, random seed
    
    Returns
    -------
    op : DataFrame, created K-folds cross-validation indices with two columns: 
        group: int, group index from 0 to k-1
        x: sample index
    '''
    np.random.seed(seed)
    op = np.random.permutation(np.arange(len(x))) % k
    return(op)


def CVdata(df, ic_x=[], ic_y=[], ir=None, k=10, f_norm_y=lambda x: x, **kwargs):
    '''
    Create a dict of cross-validation dataset for models
    
    Parameters
    ----------
    df : DataFrame, data to clean
    ictype : Series, types of columns for converting data, it should include at least one variable as "Y" 
    ir : list, index of sub-sample
    f_cv : function, use to create cross-validation indices
    
    Returns
    -------
    op : dict, including following three keys:
        X : DataFrame, independent variables
        Y : DataFrame, depedent variable(s)
        irts : Series, cross-validation group index
    '''
    if(ir is None):
        ir = df.index
    op = {"X": CLdata(df.loc[ir].reindex(columns=ic_x, fill_value=0), **kwargs), 
          "Y": f_norm_y(df.loc[ir, ic_y]),
          "irts": Kfolds(ir, k)}
    return(op)


def CVset(X, Y, irts, ig=0, ictL=None, **kwargs):
    '''
    Create a training-validation set from X, Y and cross-validation indices
    
    Parameters
    ----------
    X : DataFrame, indepedent variables
    Y : DataFrame, dependent variable(s)
    irts : Series, cross-validation indices
    ig : int, fold to use as validation index, and others as training index
    ictL : DataFrame, X columns bool indcators for each set in CV
    
    Returns
    -------
    op : dict, including following four keys:
        xt : DataFrame, training X set
        xv : DataFrame, validation X set
        yt : DataFrame, training Y set
        yv : DataFrame, validation Y set
    '''
    irv = irts == ig
    op = {"yt": Y.loc[~irv], "yv": Y.loc[irv]}
    if(ictL is None):
        op.update({"xt": X.loc[~irv], "xv": X.loc[irv]})
    else:
        ict = ictL.iloc[:, ig]
        op.update({"xt": X.loc[irt, ict], "xv": X.loc[irv, ict]})
    return(op)


def CVply(f, irts, parcv={}, f_con=list, argcon = {}, **kwargs):
    '''
    Apply function on cross-validation data sets
    
    Parameters
    ----------
    f : function, to apply cross-validation arguments
    irts : Series, cross-validation group index
    parcv : dict, keys -- argument names of f to iterate, values -- argument names of kwargs to iterate
    f_con : function, to concat iterated outputs of f
    argcon : dict, arguments of f_con
    **kwargs : arguments of f
    
    Returns
    -------
    op : list, iterated outputs of f
    '''
    parcv = dict([[x, kwargs.pop(parcv[x])] for x in parcv.keys()])
    op = f_con(map(lambda i: f(seed=i, **CVset(ig=i, irts = irts, **kwargs), **dict([[x, parcv[x][i]] for x in parcv.keys()]), **kwargs), np.unique(irts)), **argcon)
    return(op)


def CVmodel(md):
    '''
    Do cross validation given a model dictionary
    
    Parameters
    ----------
    md: dict, model dictionary, should include at least the following keys:
        X : DataFrame, indepedent variables
        Y : DataFrame, dependent variable(s)
        irts : Series, cross-validation group index
        f_model : function, model to train
        par_model : dict, arguments to train f_model
        f_loss (optional) : function, use to calculate loss, if not included, use the default option of function Loss
    
    Returns
    -------
    op : dict, updated model dictionary, with the following keys added:
        modelL : list, trained cross-validation models
        wL : DataFrame, variable weights of the cross-validation models
        yvpL : Series, predicted Ys on validation sets by different CV models
        lossL : Series, loss on validation sets
    '''
    md["modelL"] = CVply(f=MDtrain, **md)
    md["wL"] = CVply(f=MDweight, parcv={"model": "modelL"}, f_con=pd.concat, argcon = {"axis": 1, "keys": np.unique(md["irts"])}, **md)
    md["yvpL"] = CVply(f=MDpred, parcv={"model": "modelL"}, **md)
    md["lossL"] = CVply(f=Loss, parcv={"yp": "yvpL"}, f_con=pd.Series, **md)
    md["yvpL"] = pd.concat(md["yvpL"]).loc[md["Y"].index]
    return(md)


def CVweight(wL, family="normal", **kwargs):
    '''
    Analysis and test variable weights of cross-validation models
    
    Parameters
    ----------
    wL : DataFrame, variable weights of the cross-validation models
    family : str, distribution types of variable weights
    
    Returns
    -------
    op : DataFrame, the columns are Mean, Std, P-value, LowerCI and UpperCI of variable weights
    '''
    if(family == "normal"):
        op = pd.concat([wL.mean(axis=1), wL.std(axis=1)], axis=1, keys=["Mean", "Std"])
        sd = op["Std"]*np.sqrt(wL.shape[1])
        op["P-value"] = 2*st.t.cdf(-np.abs(op["Mean"]/sd), wL.shape[1]-1)
    elif(family == "binomial"):
        op = wL.sum(axis=1)
        sd = wL.std(axis=1)/op.mean()
        op = pd.concat([op/op.mean(), sd], axis=1, keys=["Mean", "Std"])
        op["P-value"] = st.norm.cdf(-(op["Mean"] - 1)/op["Std"])
    op = op.assign(LowerCI=op["Mean"]-1.96*sd, UpperCI=op["Mean"]+1.96*sd)
    return(op)

def cross_join(left, right, **kwargs):
    return(pd.merge(left.assign(_key=1), right.assign(_key=1), on="_key", **kwargs).drop("_key", axis=1))

def MCVtest(df, ictypeL, mdpar, **kwargs):
    '''
    Using different sets of variables to build cross-validation models on the same data
    
    Parameters
    ----------
    df : DataFrame, data to clean and build model
    ictypeL : list, of types of columns for converting data
    mdpar : dict, model parameter dictionary, should include at least the following keys:
        f_model : function, model to train
        par_model : dict, arguments to train f_model
        f_loss (optional) : function, use to calculate loss, if not included, use the default option of function Loss
    **kwargs : arguments of function CVdata
    
    Returns
    -------
    op : list, of model dictionaries
    '''
    op = [CVmodel({**CVdata(df, i, **kwargs), **mdpar}) for i in ictypeL]
    return(op)


def MCVoffsetmodel(mdL, X, Y, irts, mdpar, f_loss=met.roc_auc_score):
    '''
    Create a training-validation set from X, Y and cross-validation indices
    
    Parameters
    ----------
    mdL : list, of CV models
    X : DataFrame, covariates to adjust as offset
    Y : DataFrame, dependent variable(s)
    irts : Series, cross-validation indices
    mdpar : dict, model parameter dictionary for offset, should include at least the following keys:
        f_model : function, model to train
        par_model : dict, arguments to train f_model
        f_loss (optional) : function, use to calculate loss, if not included, use the default option of function Loss
    f_loss : function, use to calculate loss, if not included, use the default option of function Loss
    
    Returns
    -------
    op : DataFrame, mean loss of CV models
    '''
    op = []
    model_offset = MDtrain(xt = X, yt = Y, xv = X, yv = Y, **mdpar)
    X_offset = MDpred(model_offset, X, logit = True).rename(columns={0: "covar"})
    op.append(np.append(np.mean([[Loss(Y.loc[irts==i], j, f_loss) for j in [np.repeat(Y.loc[irts!=i].mean(), sum(irts==i)), X_offset.loc[irts==i]]] for i in np.unique(irts)], axis = 0), 0))
    print(op[-1])
    for md in mdL:
        md_h1 = CVmodel({"X": md["yvpL"].join(X_offset), "Y": Y, "irts": irts, **mdpar, "ic_offset": "covar"})
        op.append([md["lossL"].mean(), md_h1["lossL"].mean(), md_h1["wL"].loc[0].groupby(level = 1).mean()["coef"]])
        print(op[-1])
    op = pd.DataFrame(op, columns=["loss", "loss with offset", "adjusted coef"])
    return(op)


def MMCVmodel(mdsetL, mdparL):
    '''
    Parameters
    ----------
    mdsetL : list, each element is a model dictionary of datasets, including three keys:
        X : DataFrame, indepedent variables
        Y : DataFrame, dependent variable(s)
        irts : Series, cross-validation group index
    mdparL : list, each element is a model dictionary of functions and parameters, including following keys:
        namemd : str, custom name of model
        f_model : function, model to train
        par_model : dict, arguments to train f_model
        f_loss (optional) : function, use to calculate loss, if not included, use the default option of function Loss
    
    Returns
    -------
    mdLL : list (datasets) of lists (models), each element is an updated model dictionary, with the following keys added:
        modelL : list, trained cross-validation models
        wL : DataFrame, variable weights of the cross-validation models
        yvpL : list, predicted Ys on validation sets
        lossL : Series, loss on validation sets
    '''
    mdLL = [[CVmodel({**i, **j}) for j in mdparL] for i in mdsetL]
    return(mdLL)

def DoubleWeightedTstat(x, xbins = np.linspace(-10, 10, 200), distcdf = st.laplace.cdf, bounds = [(0,1), (0.1,10)], tol = 1e-7, seed = 1, plot = True, **kwargs):
    '''
    Adjust t-statistics of a large number of independent tests with Bayesian inference.
    Prior assumption: 
        1. (1-alpha) rates as zero effect, with likelihood Normal(0,1)
        2. alpha rates as non-zero effect mu, with likelihood Normal(mu,1), mu has a given prior distcdf scaled by sigma
        3. alpha and sigma are estimated by maximum a posteriori probability (MAP)
    
    Parameters
    ----------
    x : list-like, t-statistics to adjust
    xbins : array, bin edges for cutting x
    distcdf : function, cumulative distribution function of prior of non-zero t-statistics
    bounds : list, each element is tuple (min, max) bounds for parameters
    tol : float, convergence threshold
    seed : int, random seed
    plot : bool, whether plot estimation result
    **kwargs : arguments of function opt.differential_evolution

    Returns
    -------
    xadj : array, adjusted x
    '''
    ntable = np.bincount(pd.Series(pd.cut(x, xbins)).cat.codes.values, minlength = len(xbins)-1)/len(x)
    xtable = (xbins[1:] + xbins[:-1])/2
    g0 = np.exp(-xtable**2/2)
    G = np.exp(-np.subtract.outer(xtable, xtable)**2/2)
    def loss(a):
        return(-2*ntable@np.log(a[0]*(G@np.diff(distcdf(xbins/a[1])) - g0) + g0))
    model = opt.differential_evolution(loss, bounds=bounds, seed = seed, tol = tol, **kwargs)
    print(model)
    alpha, sigma = model.x
    nhat = (alpha*G@np.diff(distcdf(xbins/sigma)) + (1-alpha)*g0)
    nhat = nhat/sum(nhat)
    muhat = alpha*np.diff(distcdf(xbins/sigma)) + (1-alpha)*np.diff(st.uniform.cdf(xbins*100))
    muhat = muhat/sum(muhat)
    pdftable = muhat[:,np.newaxis]*G
    pdftable = np.divide(pdftable, np.sum(pdftable, axis = 0))
    Etable = (xtable[:,np.newaxis]*pdftable).sum(axis = 0)
    xadj = intp.interp1d(xtable, Etable)(x)
    if(plot):
        nadjtable = np.bincount(pd.Series(pd.cut(xadj, xbins)).cat.codes.values, minlength = len(xbins)-1)/len(x)
        pd.DataFrame(np.array([ntable, muhat, nhat, nadjtable]).T, index = xtable, columns = ["Count", "PriorPDF", "PostPDF", "Adjusted"]).rename_axis("T-stat").plot(logy = True, ylim = [0.5/len(x), 1])
        pd.DataFrame(np.array([1-pdftable[np.argmin(np.abs(xtable))], Etable.round(8)/xtable]).T, index = xtable, columns = ["P(Non-zero)", "Shrinkage Ratio"]).rename_axis("T-stat").plot(ylim = [0, 1])
    return(xadj)


def DWadjust_plink(daw, fw = lambda x: x["BETA"], **kwargs):
    op = daw.assign(BETA = fw, Tstat_adj = pd.Series(DoubleWeightedTstat(daw["STAT"].dropna(), **kwargs), index = daw["STAT"].dropna().index))
    op["Beta_adj"] = op["BETA"]*op["Tstat_adj"]/op["STAT"]
    return(op)