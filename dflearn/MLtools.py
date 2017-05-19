import time
import re
import inspect
import multiprocessing

import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.optimize as opt
import scipy.interpolate as intp
import patsy as ps
import statsmodels.api as sm
import sklearn.metrics as met
import sklearn.linear_model as lm
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin


def validpar(func, x):
    '''
    Avoid invalid parameters for a function.
    
    Parameters
    ----------
    func : function
        Which cannot accept invalid parameters.
    x : dict
        Including pairs of parameter name and value.
    
    Returns
    -------
    op : dict
        Valid parameters for the function.
    '''
    op = dict(i for i in x.items() if i[0] in inspect.signature(func).parameters.keys())
    return(op)


def strnum(x, f_reduce=max):
    '''
    Convert string to number, only keep numerical part (0-9 or .), otherwise return np.nan.
    
    Parameters
    ----------
    x : string
        To convert to numbers.
    f_reduce : function
        Function to reduce multiple numbers.
    
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

    
def plyargs(func, argL, argname, f_concat=list, argcon={}, **kwargs):
    '''
    Apply function on a list of arguments.
    
    Parameters
    ----------
    func : function
        To apply arguments.
    argL : list (of lists)
        Iterated argument values.
    argname : list
        Argument names, same length as argL.
    f_concat : function
        To concat iterated outputs of the function.
    argcon : dict
        arguments of f_con
    Additional keyword arguments will be passed as keywords to the function.
    
    Returns
    -------
    op : list-like
        iterated outputs of the function.
    '''
    op = f_concat(map(lambda arg: func(**dict(zip(*[argname, arg])), **kwargs), zip(*argL)), **argcon)
    return(op)


def apply_df(df, func, axis=0, n_jobs=1, **kwargs):
    '''
    Apply function on each row/column of dataframe and return a Series.
    
    Parameters
    ----------
    df : DataFrame
        To apply function.
    func : function
        Function with input (index, Series) to apply to each column/row.
    axis :  {0, 1}, default 0
        - 0 or 'index': apply function along rows.
        - 1 or 'columns': apply function along columns.
    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel.
    Additional keyword arguments will be passed as keywords to the function.
    
    Returns
    -------
    S : list-like
        iterated outputs of f.
    '''
    if axis == 0:
        pairs = df.iteritems()
        index = df.columns
    else:
        pairs = df.iterrows()
        index = df.index
    S = pd.Series([func(i_val) for i, i_val in pairs], index, **kwargs)
    return(S)

    
def cross_join(left, right):
    '''
    Cross outer join of two DataFrames along column.
    
    Parameters
    ----------
    left : DataFrame
    right : DataFrame
    
    Returns
    -------
    DataFrame
        
    '''
    return(pd.merge(left.assign(_key=1), right.assign(_key=1), on="_key").drop("_key", axis=1))
    
    
def collinearvif(df):
    '''
    Compute variance inflation factor (VIF) of a DataFrame.
    
    Parameters
    ----------
    df : DataFrame
    
    Returns
    -------
    op : Series
        VIF values of each column.
    '''
    op = pd.Series(np.diag(np.linalg.inv(df.corr())), df.columns, name="CollinearVIF")
    return(op)


def summary(df, n=5, pct=[0.1, 0.5, 0.9]):
    '''
    Summarize DataFrame along columns for data type, sample size, numerical statistics and frequency
    
    Parameters
    ----------
    df : DataFrame
        To summary.
    n : int
        The number of foremost frequent categories of frequency table.
    pct : list
        Percentiles for numerical statistics.

    Returns
    -------
    op : DataFrame
        summary of DataFrame along columns.
    '''
    
    def freq(s):
        op = pd.value_counts(s)
        op = pd.concat([pd.Series(op.index[:n]).rename(lambda x: "FreqCat{}".format(x+1)), 
                        pd.Series(op.values[:n]).rename(lambda x: "FreqVal{}".format(x+1)).T, 
                        pd.Series(op.iloc[n:].sum(), index=["Freq_Others"])])
        return(op)
    op = pd.concat([df.dtypes.rename("Type"), 
                    df.notnull().sum().rename("N"), 
                    df.describe(pct).iloc[1:].T, 
                    df.apply(freq).T], axis=1).loc[df.columns]
    return(op)


## Clean Data

def CLtimenum(s):
    '''Convert datetime variable to numeric variable.'''
    op = pd.DatetimeIndex(s)
    op = pd.get_dummies(pd.DataFrame(dict(zip(*[["{}__{}".format(s.name, i) for i in ["Month", "Year"]], [op.month, op.year]])), index=s.index).astype("O"))
    return(op)


def CLnormal(df):
    '''
    Normalize the distribution of numeric DataFrame to standard normal distribution by rank.
    
    See also
    --------
    CLscale
    '''
    op = pd.DataFrame(st.norm.ppf((df.rank(pct = True) - 0.5/df.shape[0]).fillna(0.5)), index=df.index, columns=df.columns)
    return(op)


def CLscale(df):
    '''
    Standardize the distribution of numeric DataFrame to mean 0 and standard deviation 1.
    
    See also
    --------
    CLnormal
    '''
    op = ((df - df.mean())/df.std()).fillna(0)
    return(op)


def CLsparse_cat(s, sp=0.01):
    '''
    Clean a object Series with sparse values converted to "others".
    '''
    sfreq = pd.value_counts(s) > sp*s.shape[0]
    op = s.where(s.isin(sfreq.index[sfreq]), "others")
    return(op)


def CLsparse(s):
    '''
    Check the percentage of non-most-frequent and non-null values in a Series.
    '''
    op = 1 - s.isnull().mean()
    if op > 0:
        op -= pd.value_counts(s).iloc[0]/s.shape[0]
    return(op)


def CLdata(df, sp=0, cor=1, f_norm=CLscale, formula=None, **kwargs):
    '''
    Clean a pandas DataFrame according to dtype (numeric or object)
    
    Parameters
    ----------
    df : DataFrame
        data to clean.
    sp : float
        between [0, 1), threshold of ratio of unique and missing values to delete variables.
    cor : float
        between (0, 1], threshold of absolute correlation to remove later collinear variables.
    f_norm : function
        use to standardize numeric variables.
    formula : str
        add transformed or interaction terms of variables.

    
    Returns
    -------
    df_clean : DataFrame
        cleaned data
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

## Machine Learning Procedures

def MDinit(f_model=lm.LogisticRegression, par_model={}, random_state=0, **kwargs):
    '''
    Initiate a model of scikit-learn form.
    
    Parameters
    ----------
    random_state : int or list
        the seed used by the random number generator. If a list, create a list of models.
    f_model : function
        model to train.
    par_model : dict
        arguments to initiate a f_model.
    
    Returns
    -------
    out: initiated model or model list
    '''
    if hasattr(random_state, "__iter__"):
        return([MDinit(f_model, par_model, i) for i in random_state])
    else:
        par = {'learning_rate': 0.05, 'n_jobs': -1, "class_weight": 'balanced'}
        par.update(par_model)
        return(f_model(**validpar(f_model, {**par_model, "random_state": random_state, "random_state": random_state})))
    
    
def MDfit(model, xt, yt, xv=None, yv=None, par_fit={'verbose': False}, **kwargs):
    '''
    Train a model of scikit-learn form
    
    Parameters
    ----------
    xt : DataFrame
        training X set.
    xv : DataFrame
        validation X set.
    yt : DataFrame
        training Y set.
    yv : DataFrame
        validation Y set.
    random_state : int
        the seed used by the random number generator.
    model : model to train
    par_fit : dict
        arguments to train f_model.
    
    Returns
    -------
    model : trained model
    '''
    par = {"eval_set": [(xv, yv.iloc[:,0])], 'verbose': False}
    par.update(par_fit)
    model.fit(xt, yt.iloc[:,0], **validpar(model.fit, par))
    return(model)


def MDpred(model, xv, ic_offset=[], f_loss=met.roc_auc_score, logit=False, **kwargs):
    '''
    Use trained model and validation X to predict Y.
    
    Parameters
    ----------
    model : trained model
    xv : DataFrame
        validation X to predict Y.
    ic_offset : str
        offset column name of X for Y.
    
    Returns
    -------
    yvp : DataFrame
        predicted Y for validation set.
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
    Use trained model and X to get variable weights.
    
    Parameters
    ----------
    model : trained model
    xt : DataFrame
        get variable name from training X set.
    
    Returns
    -------
    op : Series
        variable weights of the model.
    '''
    if type(model).__name__ in ['Ridge', "Lasso"]:
        op = pd.Series(model.coef_, index=xt.drop(ic_offset, axis=1).columns)
    elif type(model).__name__ in ['LogisticRegression']:
        op = pd.Series(model.coef_[0], index=xt.drop(ic_offset, axis=1).columns)
    elif 'sklearn.ensemble' in type(model).__module__:
        op = pd.Series(model.feature_importances_, index=xt.drop(ic_offset, axis=1).columns)
    elif type(model).__name__ in ["GLMResultsWrapper"]:
        op = pd.concat([model.params, model.t()], axis=1, keys=["coef", "t"])
    else:
        op = pd.Series()
    return(op)


def Loss(yv, yp, f_loss=met.roc_auc_score, **kwargs):
    '''
    Get loss between true validation Y and predicted Y
    
    Parameters
    ----------
    yv : DataFrame
        validation Y.
    yp : DataFrame
        predicted Y.
    f_loss : function
        use to calculate loss.
    
    Returns
    -------
    op : float
        loss
    '''
    op = f_loss(np.array(yv).flatten(), np.array(yp).flatten())
    return(op)


def roc(yv, yp, plot=False, **kwargs):
    '''
    Get ROC curve between binary validation Y and predicted Y.
    
    Parameters
    ----------
    yv : DataFrame
        binary validation Y.
    yp : DataFrame
        predicted Y.
    plot : bool
        whether plot a ROC curve.
    
    Returns
    -------
    op : DataFrame
        elements of an ROC curve including: true positive ratio, false positive ratio, and threshold.
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


def DMinit(mdset_df, mdpar_df):
    '''
    Initiate a DataFrame-based CV model set.
    
    Parameters
    ----------
    mdset_df : DataFrame
        for CV data.
    mdpar_df : DataFrame
        for model parameters.
    
    Returns
    -------
    out: DataFrame
        cross-joined CV data-model list.
    '''
    md_df = cross_join(mdset_df, mdpar_df)
    md_df['random_state'] = md_df['irts'].apply(np.unique)
    md_df["modelL"] = md_df.apply(lambda x: MDinit(**x.to_dict()), axis=1)
    return(md_df.drop(["random_state", "f_model", "par_model"], axis=1))


def Kfolds(x, k=10, random_state=1):
    '''
    Use index to create a list of K-folds cross-validation indices.
    
    Parameters
    ----------
    x : list
        sample index.
    k : int
        number of folds.
    random_state : int
        the seed used by the random number generator.
    
    Returns
    -------
    op : array
    '''
    np.random.seed(random_state)
    op = np.random.permutation(np.arange(len(x))) % k
    return(op)


def CVdata(df, ic_x=[], ic_y=[], ir=None, k=10, f_norm_y=lambda x: x, random_state=0, **kwargs):
    '''
    Create a dict of cross-validation dataset for models
    
    Parameters
    ----------
    df : DataFrame
        data to clean
    ic_x : list
        column names used as X
    ic_y : list
        column names used as Y
    ir : list
        index of sub-sample
    k : int
        number of cross-validation folds
    
    Returns
    -------
    op : dict, including following three keys:
        X : DataFrame, independent variables
        Y : DataFrame, depedent variable(s)
        irts : 1-d array, cross-validation group index
    '''
    if(ir is None):
        ir = df.index
    op = {"X": CLdata(df.loc[ir].reindex(columns=ic_x, fill_value=0), **kwargs), 
          "Y": f_norm_y(df.loc[ir, ic_y]),
          "irts": Kfolds(x=ir, k=k, random_state=random_state)}
    return(op)


def CVset(X, Y, irts, ig=0, ictL=None, **kwargs):
    '''
    Create a training-validation set from X, Y and cross-validation indices
    
    Parameters
    ----------
    X : DataFrame
        indepedent variables
    Y : DataFrame
        dependent variable(s)
    irts : list-like
        cross-validation indices
    ig : int
        fold to use as validation index, and others as training index
    ictL : DataFrame
        X columns bool indcators for each set in CV
    
    Returns
    -------
    op : dict
        including following four keys:
        xt : DataFrame
            training X set
        xv : DataFrame
            validation X set
        yt : DataFrame
            training Y set
        yv : DataFrame
            validation Y set
    '''
    irv = irts == ig
    op = {"yt": Y.loc[~irv], "yv": Y.loc[irv]}
    if(ictL is None):
        op.update({"xt": X.loc[~irv], "xv": X.loc[irv]})
    else:
        ict = ictL.iloc[:, ig]
        op.update({"xt": X.loc[irt, ict], "xv": X.loc[irv, ict]})
    return(op)


def CVset_df(X, Y, irts, ig=None, **kwargs):
    '''
    Create a DataFrame-form training-validation set from X, Y and cross-validation indices
    
    Parameters
    ----------
    X : DataFrame
        indepedent variables
    Y : DataFrame
        dependent variable(s)
    irts : list-like
        cross-validation indices
    ig : int
        fold to use as validation index, and others as training index. if None, use all folds
    
    Returns
    -------
    op : DataFrame
        columns including following four keys:
        xt : DataFrame
            training X set
        xv : DataFrame
            validation X set
        yt : DataFrame
            training Y set
        yv : DataFrame
            validation Y set
    '''
    if ig is None:
        return(pd.DataFrame.from_dict([CVset(X=X, Y=Y, irts=irts, ig=i, **kwargs) for i in np.unique(irts)]))
    else:
        return(pd.DataFrame.from_dict([CVset(X=X, Y=Y, irts=irts, ig=ig, **kwargs)]))

    
def CVply(func, irts, parcv={}, f_con=list, argcon={}, **kwargs):
    '''
    Apply function on cross-validation data sets
    
    Parameters
    ----------
    func : function
        to apply cross-validation arguments
    irts : Series
        cross-validation group index
    parcv : dict
        - keys: argument names of the function to iterate
        - values: argument names of kwargs to iterate
    f_con : function
        to concat iterated outputs of the function
    argcon : dict
        arguments of f_con
    Additional keyword arguments will be passed as keywords to the function.
    
    Returns
    -------
    op : list
        iterated outputs of the function
    '''
    parcv = dict([[x, kwargs.pop(parcv[x])] for x in parcv.keys()])
    op = f_con(map(lambda i: func(random_state=i, **CVset(ig=i, irts=irts, **kwargs), **dict([[x, parcv[x][i]] for x in parcv.keys()]), **kwargs), np.unique(irts)), **argcon)
    return(op)


def CVweight(wL, family="normal", **kwargs):
    '''
    Analysis and test variable weights of cross-validation models
    
    Parameters
    ----------
    wL : DataFrame
        variable weights of the cross-validation models
    family : str
        distribution types of variable weights
    
    Returns
    -------
    op : DataFrame
        the columns are Mean, Std, P-value, LowerCI and UpperCI of variable weights
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


class LinearClass(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    def summary(self):
        model_t = st.t(df=self.dof)
        out_S = pd.Series([self.sigma2, len(self.w), self.dof], ["Total Sigma2", "Parameters", "Degree of freedom"])
        out_df = pd.DataFrame({"coef": self.w, "se": self.w_se}, self.columns)
        out_df["t"] = out_df["coef"]/out_df["se"]
        out_df["p-value"] = 2*model_t.cdf(-np.abs(out_df["t"]))
        out_df["0.025 CI"] = out_df["coef"]+model_t.ppf(0.025)*out_df["se"]
        out_df["0.975 CI"] = out_df["coef"]+model_t.ppf(0.975)*out_df["se"]
        return(out_S, out_df)
    
    
class LinearSingleModel(LinearClass):
    def __init__(self):
        pass
    def fit(self, X, Y, X_offset=None):
        self.columns = X.columns
        if X_offset is not None:
            X0 = np.hstack([np.ones([Y.shape[0], 1]), X_offset.values])
        else:
            X0 = np.ones([Y.shape[0], 1])
        Y = Y.values
        X = X.values
        XX_A = X0.T @ X0
        XX_B = X0.T @ X
        XX_c = (X ** 2).sum(axis=0)
        XY_a = X0.T @ Y
        XY_b = X.T @ Y
        XX_Ainv = np.linalg.inv(XX_A)
        XX_AinvB = XX_Ainv @ XX_B
        self.w0 = XX_Ainv @ XY_a
        self.dof = len(X) - len(self.w0) - 1
        self.sigma2 = ((Y - X0 @ self.w0)**2).sum()/self.dof
        XXinv_c = 1/(XX_c - (XX_B * XX_AinvB).sum(axis=0))
        XXinv_B = - XXinv_c * XX_AinvB
        # XXinv_Adiag = np.diag(XX_Ainv)[:, np.newaxis] + XXinv_c * (XX_AinvB ** 2)
        # self.W0 = self.w0 + XXinv_c * XX_AinvB * (XX_AinvB.T @ XY_a)[:,0] + XXinv_B * XY_b.T
        self.w = (XY_a.T @ XXinv_B + XXinv_c * XY_b.T)[0]
        self.w_se = np.sqrt(XXinv_c * self.sigma2)
        return(self)

    
class LinearMixedModel(LinearClass):
    
    def __init__(self):
        self.G = None
        
    def fit_kernel(self, X_g, w_g=None, scale=False):
        '''
        Create standardized kernel for linear mixed model

        Parameters
        ----------
        X_g : DataFrame
            of shape (n, p), random effects matrix
        w_g : 1d array
            of shape (p,), scale of columns. 
        scale : bool
            whether scale columns with mean 0 and std 1
    
        Attributes
        ----------
        G : 2d array
            of shape (n, n), kernel of random effects in linear mixed model
        '''
        if scale:
            X_g = CLscale(X_g) / np.sqrt(X_g.shape[1]) 
        if w_g is None:
            self.G = X_g.dot(X_g.T)
        else:
            X_g *= w_g
            self.G = X_g.dot(X_g.T) / (w_g ** 2).mean()
        return(self)

    def fit_coef(self, h2):
        D = 1 + h2 * (self.S - 1)
        XDX = (self.UX.T / D) @ self.UX
        XDY = (self.UX.T / D) @ self.UY
        self.w = np.linalg.solve(XDX, XDY)
        self.sigma2 = ((self.UY - self.UX @ self.w).T ** 2 / D).mean()
        return((np.log(D*self.sigma2).sum() + np.linalg.slogdet(XDX/self.sigma2)[1] - np.linalg.slogdet(self.XX)[1]) / len(self.S))
        
    def fit(self, X, Y, **kwargs):
        '''
        Factored spectrally transformed linear mixed models

        Parameters
        ----------
        X : DataFrame
            of shape (n, p), covariate matrix
        Y : DataFrame
            of shape (n, 1), response variable
    
        Attributes
        ----------
        w : coefficients for X
        sigma2 : total variance of residuals
        h2 : heritability of random effects
        loss : -2 RE log-likelihood divided by n
        '''
        self.columns = np.insert(X.columns, 0, "(Intercept)")
        X_c = np.hstack([np.ones([X.shape[0], 1]), X.values])
        if self.G is None:
            self.S, U = np.ones(len(Y)), np.diag(np.ones(len(Y)))
        else:
            self.S, U = np.linalg.eigh(self.G.loc[Y.index].values)
        self.dof = X.shape[0] - X.shape[1] - 1
        self.UX = U.T @ X_c
        self.UY = U.T @ Y.values
        self.XX = X_c.T @ X_c
        if self.G is None:
            self.h2 = 0
        else:
            model = opt.differential_evolution(self.fit_coef, bounds=[(0,1)], tol=1e-7, **kwargs)
            self.h2 = model.x[0]
            self.dof -= 1
        self.loss = self.fit_coef(self.h2)
        self.w = self.w.flatten()
        self.w_se = np.sqrt(np.diag(np.linalg.inv(self.XX))*(1-self.h2)*self.sigma2)
        return(self)
    
    def predict(self, X):
        return(self.w.values[0]+X.dot(self.w.values[1:]))

    
class LDBayesCpiTstat(BaseEstimator, TransformerMixin):
    '''
    Bayes C-pi adjustment of single-association t-statistics with LD score and heritability.
    '''
    def __init__(self, width=8, num=400):
        self.xbins = np.linspace(-2 * width, 2 * width, 2 * num)
        self.xtable = (self.xbins[1:] + self.xbins[:-1]) / 2
        self.G = np.exp(- np.subtract.outer(self.xtable, self.xtable) ** 2 / 2)/np.sqrt(2 * np.pi)
        self.g0 = np.exp(- self.xtable ** 2 / 2)/np.sqrt(2 * np.pi)
        self.delta = 2 * width / (num - 1)
        self.xlim = [- width, width]

    def diff_d(self, sigma):
        return(np.diff(self.distcdf(self.xbins/sigma)))
        
    def pdf_x(self, pi, sigma):
        return((1 - pi) * self.g0 + pi * self.G @ self.diff_d(sigma))
    
    def diff_mu(self, pi, sigma):
        return((1 - pi) * np.diff(st.uniform.cdf(self.xbins * 100)) + pi * self.diff_d(sigma))
    
    def cut_bins(self, x, x_bins):
        return(pd.Series(pd.cut(x, x_bins, right=False)).cat.codes.values)
    
    def count(self, x):
        return(np.bincount(self.cut_bins(x, self.xbins), minlength=len(self.xbins) - 1))
        
    def loss(self, z):
        xpdf_table = np.vstack([self.pdf_x(z[0], z[1] * c) for c in self.c_table])
        return(-2 * (self.x_count * np.log(xpdf_table)).sum())
    
    def fit(self, x, ldsc=None, h2=0, n=0, distcdf=st.laplace.cdf, bounds=[(0, 1), (0.1, 10)], tol=1e-4, plot=True, **kwargs):
        """
        Parameters
        ----------
        x : list-like
            t-statistics to adjust
        ldsc : list-like
            LD scores with the same dimension of x
        h2 : float
            heritability of phenotype, between 0 and 1
        n : int
            sample size
        distcdf : function
            cumulative distribution function of prior of non-zero t-statistics
        tol : float
            log-likelihood convergence threshold
        plot : bool
            whether plot estimation result
        Additional keyword arguments will be passed as keywords to opt.differential_evolution
        """
        if ldsc is None:
            ldsc = np.ones(len(x))
        self.n = n
        self.h2 = h2
        self.p = len(x)
        self.distcdf = distcdf
        self.ldsc_sigma = self.h2 * self.n / ((1 - self.h2) * self.p)
        self.ldsc_mean = np.mean(ldsc)
        self.ldsc_max = np.max(ldsc)
        self.c_min = 1 / np.sqrt(1 + self.ldsc_max * self.ldsc_sigma)
        self.c_mean = 1 / np.sqrt(1 + self.ldsc_mean * self.ldsc_sigma)
        self.c_table = np.arange(1, max(self.c_min - 0.04, 0), -0.02)
        self.c_bins = np.arange(1.01, max(self.c_min - 0.06, -0.01), -0.02)
        self.ldsc_table = 1 / self.c_table ** 2 - 1
        self.ldsc_bins = 1 / self.c_bins ** 2 - 1

        group = self.cut_bins(ldsc * self.ldsc_sigma, self.ldsc_bins)
        self.x_count = np.vstack([self.count(x[group==i]) for i in range(len(self.ldsc_table))])
        self.nullpdf = self.pdf_x(0, 1e-7)
        self.model_fit = opt.differential_evolution(self.loss, bounds=bounds, tol=tol, **kwargs)
        self.pi, self.sigma = self.model_fit.x
        self.adjuster(self.pi, self.sigma, plot=plot)
        return(self)
    
    def adjuster_uni(self, pi, sigma):
        xpdf = self.pdf_x(pi, sigma)
        mudiff = self.diff_mu(pi, sigma)
        ddiff = self.diff_d(sigma)
        Px = (self.G / xpdf).T
        Ex = pi * Px @ (self.xtable * ddiff)
        Pmu = self.delta * (self.G.T / xpdf) @ self.G
        Emu = pi * Pmu @ (self.xtable * ddiff)
        return(Ex, Emu)

    def adjuster(self, pi, sigma, plot=True):
        Ex_table, Emu_table = zip(*[self.adjuster_uni(pi, c * sigma) for c in self.c_table])
        Ex_table = np.vstack(Ex_table).T
        Emu_table = np.vstack(Emu_table).T
        self.adjuster_x = intp.RectBivariateSpline(self.xtable, self.ldsc_table, Ex_table, kx=1, ky=1)
        self.adjuster_mu = intp.RectBivariateSpline(self.xtable, self.ldsc_table, Emu_table, kx=1, ky=1)
        self.xpdf = self.pdf_x(pi, sigma * self.c_mean)
        self.mudiff = self.diff_mu(pi, sigma * self.c_mean)
        self.ddiff = self.diff_d(sigma * self.c_mean)
        Px = (self.G / self.xpdf).T
        self.Ex = pi * Px @ (self.xtable * self.ddiff)
        self.Vx = pi * Px @ (self.xtable ** 2 * self.ddiff) - self.Ex ** 2
        Pmu = self.delta * (self.G.T / self.xpdf) @ self.G
        self.Emu = pi * Pmu @ (self.xtable * self.ddiff)
        self.Vmu = pi * Pmu @ (self.xtable ** 2 * self.ddiff) - self.Emu**2
        self.mucdf = self.mudiff.cumsum()
        self.cdf_mu = intp.interp1d(self.xbins[1:], self.mucdf)
        self.deviance_null = self.loss([0, 1e-7]) - self.p * (np.log(2 * np.pi) + 1)
        self.deviance = self.loss([self.pi, self.sigma]) - self.p * (np.log(2 * np.pi) + 1)
        if plot:
            (pd.DataFrame(np.array([1 - (1 - pi) * Px[:, np.argmin(np.abs(self.xtable))], 
                                   self.Ex / (self.xtable + 1e-8), 
                                   self.Emu / (self.xtable + 1e-8)]).T, 
                         self.xtable, ["P(non-zero)", "t-value reduce", "Effect size reduce"])
             .rename_axis("T-stat")
             .plot(ylim=[0, 1], xlim=self.xlim, grid=True, 
                   title="Inference of mu given t-value \n pi: {:.3f}, sigma: {:.3f}, mean c: {:.2f}".format(pi, sigma, self.c_mean)))
            (pd.DataFrame(np.outer(np.sqrt(self.Vx), [-2, 0, 2]), 
                          self.xtable, ["Lower", "E(mu|x)", "Upper"])
             .add(self.Ex, axis=0)
             .plot(xlim=self.xlim, ylim=self.xlim, grid=True, title="CI of mu given t-value (x)"))
            (pd.DataFrame(np.outer(np.sqrt(self.Vmu), [-2, 0, 2]), 
                          self.xtable, ["Lower", "E(mu|mu0)", "Upper"])
             .add(self.Emu, axis=0)
             .plot(xlim=self.xlim, ylim=self.xlim, grid=True, title="CI of mu given effect size (mu0)"))
        return(self)
    
    def transform(self, x, ldsc=None, plot=True):
        if ldsc is None:
            ldsc = np.ones(len(x))
        x_adj = self.adjuster_x(x, ldsc * self.ldsc_sigma, grid=False)
        w_adj = x_adj/x
        self.dof = (w_adj.sum()**2)/(w_adj**2).sum()
        if plot:
            x_adj_count = self.count(x_adj)
            (pd.DataFrame(np.array([self.x_count.sum(axis=0)/len(x), self.mudiff, self.xpdf*self.delta, x_adj_count/len(x), self.nullpdf*self.delta]).T, 
                          self.xtable, ["Count", "PriorDist", "PostDist", "Adjusted", "NullDist"]).rename_axis("T-stat")
             .plot(logy = True, ylim = [0.5/len(x), 1], xlim=self.xlim, grid=True, title="df: {}, adjusted df: {:.2f}\nNull Deviance: {:.3f}, Deviance: {:.3f}".format(len(x), self.dof, self.deviance_null, self.deviance)))
        return(x_adj)
    
    def summary(self, cdf=[0.5, 0.9, 0.95, 0.99, 0.999], mu=None, M=1, plot=False):
        self.adjuster(self.pi, self.sigma*np.sqrt(M), plot=plot)
        if mu is None:
            mu = np.array([opt.newton(lambda x: self.cdf_mu(x) - (1+i)/2, 0) for i in cdf])
        dist_var = np.sum(self.xtable**2*np.diff(self.distcdf(self.xbins)))
        dist_kurt = np.sum(self.xtable**4*np.diff(self.distcdf(self.xbins)))/dist_var**2 - 3
        out_S = pd.Series([self.n, M, self.h2, self.ldsc_mean, self.ldsc_max, self.c_mean, self.c_min, self.pi, self.sigma*np.sqrt(M), np.sqrt(self.h2*self.n/(self.p*(1-self.h2)*dist_var)), dist_var, dist_kurt, self.deviance_null, self.deviance, self.p, self.dof], ["Sample size (N)", "Sample adjust (M)", "heritability (h2)", "Mean LD", "Max LD", "Mean c", "Min c", "Effect ratio (pi)", "Effect scale (sigma)", "Derived sqrt(pi)*sigma", "Dist variance", "Dist kurtosis", "Null Deviance", "Deviance", "df", "Adjusted df"])
        out_df = pd.DataFrame({"CDF": 2*self.cdf_mu(mu)-1, "power": self.adjuster_mu(mu, self.ldsc_sigma * self.ldsc_mean, grid=False)/mu}, mu).rename_axis("effect t size")
        return(out_S, out_df)
    

## deprecated
def MLstatsmodel(xt, yt, xv=None, yv=None, random_state=0, f_model=sm.GLM, par_model={"family": sm.families.Binomial}, ic_offset=[], **kwargs):
    '''
    Train a model of statsmodel form
    
    Parameters
    ----------
    xt : DataFrame
        training X set.
    xv : DataFrame
        validation X set.
    yt : DataFrame
        training Y set.
    yv : DataFrame
        validation Y set.
    random_state : int
        the seed used by the random number generator.
    f_model : function
        model to train.
    par_model : dict
        arguments to train f_model.
    ic_offset : str
        offset column name of X of which the beta is fixed at 1.
    
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
def MLmodel(xt, yt, xv=None, yv=None, random_state=0, f_model=lm.LogisticRegression, par_model={}, par_fit={'verbose': False}, **kwargs):
    '''
    Train a model of scikit-learn form
    
    Parameters
    ----------
    xt : DataFrame, training X set
    xv : DataFrame, validation X set
    yt : DataFrame, training Y set
    yv : DataFrame, validation Y set
    random_state : int, the seed used by the random number generator
    f_model : function, model to train
    par_model : dict, arguments to train f_model
    
    Returns
    -------
    model : trained model
    '''
    par = {"random_state": random_state, "n_estimators": max(100, xt.shape[1]), 'learning_rate': 0.05, 'n_jobs': -1, "class_weight": 'balanced',
           "random_state": random_state, "colsample_bylevel": 1/(1+np.log(xt.shape[1]))}
    par.update(par_model)
    model = f_model(**validpar(f_model, par))
    model.fit(xt, yt.iloc[:,0], **validpar(model.fit, {"eval_set": [(xv, yv.iloc[:,0])], 'verbose': False, **par_fit}))
    return(model)
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
def MLsinglereg(xt, yt, xv=None, yv=None, random_state=0, par_model={}, ic_offset=[], rank=True, pct=1, **kwargs):
    '''
    Train a model of statsmodel form
    
    Parameters
    ----------
    xt : DataFrame, training X set
    xv : DataFrame, validation X set
    yt : DataFrame, training Y set
    yv : DataFrame, validation Y set
    random_state : int, the seed used by the random number generator
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
def MCVoffsetmodel(mdL, X, Y, irts, mdpar, f_loss=met.roc_auc_score):
    '''
    Create a training-validation set from X, Y and cross-validation indices
    
    Parameters
    ----------
    mdL : list
        of CV models
    X : DataFrame
        covariates to adjust as offset
    Y : DataFrame
        dependent variable(s)
    irts : Series
        cross-validation indices
    mdpar : dict
        model parameter dictionary for offset, should include at least the following keys:
        f_model : function
            model to train
        par_model : dict
            arguments to train f_model
        f_loss (optional) : function
            use to calculate loss, if not included, use the default option of function Loss
    f_loss : function
        use to calculate loss, if not included, use the default option of function Loss
    
    Returns
    -------
    op : DataFrame
        mean loss of CV models
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