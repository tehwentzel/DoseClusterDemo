import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from Constants import Const
import statsmodels.api as sm
from AucComp import delong
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif, f_regression, mutual_info_regression, mutual_info_classif
from sklearn.metrics import accuracy_score,f1_score, auc, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score, recall_score, precision_score, roc_curve
from scipy.stats import chi2_contingency, fisher_exact
import Utils
import re
from scipy.stats import chi2
import joblib

def predict_cv(model,x,y,cvsize=None):
    #currently leave-one-out
    predictions = []
    y = y.reshape(-1,1)
    if cvsize == None:
        cvsize = int(x.shape[0]*.1)+1
    nsteps = int(np.ceil(x.shape[0]/cvsize))
    start = 0
    for i in range(nsteps):
        stop = min(start + cvsize,x.shape[0])
        test_idx = np.arange(start,stop)
        x_train = np.delete(x, test_idx,axis=0)
        x_test = x[test_idx]
        y_train = np.delete(y,test_idx)
        y_test = y[test_idx]
        
        if x_test.ndim < 2:
            x_test = x_test.reshape(1,-1)
        model.fit(x_train,y_train)
        
        ypred = model.predict_proba(x_test)
        predictions.append(ypred)
        
        start=stop
    ypred = np.concatenate(predictions)
    ypred = ypred.reshape(x.shape[0],-1)
    return ypred

def get_all_dvh(ct, key='V'):
    return sorted([col for col in ct.df if re.match(key+'\d+',col) is not None], key = lambda x: int(x[1:]) )

def contingency(v1,v2):
    n_v1 = len(np.unique(v1))
    n_v2 = len(np.unique(v2))
    table = np.zeros((n_v1,n_v2))
    for i, vv1 in enumerate(np.unique(v1)):
        for ii,vv2 in enumerate(np.unique(v2)):
            in_cell = (v1 == vv1) & (v2 == vv2)
            table[i,ii] = in_cell.sum()
    return table

def boolean_fisher_exact(v1,v2):
    ctable = contingency(v1,v2)
    return fisher_exact(ctable)

def vector_chi2(x,y):
    x = x.ravel()
    y = y.ravel()
    ctable = contingency(x,y)
    res = chi2_contingency(ctable)
    return res[0], res[1]

def extract_dose_vals(df,organs,features,include_limits = False):
    oidxs = [Const.organ_list.index(o) for o in organs if o in Const.organ_list]
    df = df.copy()
    vals = []
    names = []
    for f in features:
        for (oname, oidx) in zip(organs,oidxs):
            values = df[f].apply(lambda x: x[oidx]).values
            vals.append(values.reshape((-1,1)))
            names.append(f+'_'+oname)
    vals = np.hstack(vals)
    vals = pd.DataFrame(vals,columns=names,index=df.index)
    if include_limits:
        limit_cols = [t for t in df.columns if '_limit' in t]
        for l in limit_cols:
            vals[l] = df[l].astype(int).fillna(0)
    return vals 

def get_outcomes(df,symptoms,dates,threshold=None):
    date_idxs = [i for i,d in enumerate(df.dates.iloc[0]) if d in dates]
    res = []
    get_max_sval = lambda s: df['symptoms_'+s].apply(lambda x: np.max([x[i] for i in date_idxs]) ).values
    res = {symp:get_max_sval(symp) for symp in symptoms}
    return pd.DataFrame(res,index=df.index)

def add_post_clusters(df,post_results):
    cmap = {}
    for c_entry in post_results['clusterData']:
        cId = c_entry['clusterId']
        for pid in c_entry['ids']:
            cmap[int(pid)] = cId
    df = df.copy()
    df['post_cluster'] = df.id.apply(lambda i: cmap.get(int(i),-1))
    return df
        
def process_rule_async(args):
    [df,col,y,currval,min_split_size,min_odds,min_info] = args
    vals = df[col]
    rule = vals >= currval
    entry = {
        'features': [col],
        'thresholds': [currval],
        'splits': [rule],
        'rule': rule
    }
    entry = evaluate_rule(entry,y)
    if valid_rule(entry,min_split_size,min_odds=min_odds,min_info=min_info):
        return entry
    return False
    
def get_rule_df(df,y,granularity=2,min_split_size=10,min_odds=0,min_info=.01):
    split_args = []
    minval = df.values.min().min()
    maxval = df.values.max().max()
    granularity_vals = [i*granularity + minval for i in np.arange(np.ceil(maxval/granularity))]
    for col in df.columns:
        if '_limit' in col:
            split_args.append((df,col,y,.5,1,0,0))
        else:
            for g in granularity_vals:
                split_args.append((df,col,y,g,min_split_size,min_odds,min_info))
    splits = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(process_rule_async)(args) for args in split_args)
    return [s for s in splits if s is not False]

def combine_rule(r1,r2):
    if r1 is None:
        combined = r2
    elif r2 is None:
        combined = r1
    else:
        newthresholds = r1['thresholds'][:]
        newfeatures = r1['features'][:]
        newsplits = r1['splits'][:]
        newrule = r1['rule']
        fstring = stringify_features(newfeatures)
        for i,f in enumerate(r2['features']):
            #only one split per feature
            if stringify_features([f]) not in fstring:
                newfeatures.append(f)
                t = r2['thresholds'][i]
                s = r2['splits'][i]
                newthresholds.append(t)
                newsplits.append(s)
                newrule = newrule*s
        combined = {
            'features': list(newfeatures),
            'thresholds': list(newthresholds),
            'splits': newsplits,
            'rule': newrule
        }
    return combined

def evaluate_rule(rule, y):
    r = rule['rule']
    upper = y[r]
    lower = y[~r]
    entry = {k:v for k,v in rule.items()}
    entry['info'] = mutual_info_classif(r.values.reshape(-1,1),y.values.ravel(),
                                        random_state=1,discrete_features=True,n_neighbors=5)[0]
    ucount = upper.mean().values[0]
    lcount = lower.mean().values[0]
    if ucount < lcount:
        temp = ucount
        ucount = lcount
        lcount = temp
    lcount = max(lcount, 1)
    entry['odds_ratio'] = ucount / lcount
    for prefix, yy in zip(['lower','upper'],[lower,upper]):
        entry[prefix+'_count'] = yy.shape[0]
        entry[prefix+'_tp'] = yy.sum().values[0]
        entry[prefix+'_mean'] = yy.mean().values[0]
    return entry 

def filter_rules(rulelist, bests,tholds,criteria):
    is_best = lambda r: (r[criteria] >= bests.get(stringify_features(r['features']),0)) and (
        stringify_thresholds(r['thresholds']) == tholds.get(stringify_features(r['features'])) )
    filtered = [r for r in rulelist if is_best(r)]
    return filtered

def seperate_organ(string):
    if 'max_dose' in string or 'mean_dose' in string:
        return string.replace('max_dose','').replace('mean_dose','')
    if string == 'V5' or string == 'D5':
        return string.replace('V5','').replace('D5','')
    else:
        return string[3:]
    
def stringify_features(l):
    #turns a list of features in the form 'VXX_Organ' into a hashable set
    #removes V thing becuase I think it shold be per organ
    return ''.join(sorted([seperate_organ(ll) for ll in l]))

def stringify_thresholds(t):
    return ''.join([str(int(tt)) for tt in t])

def combine_and_eval_rule(args):
    [baserule,rule,outcome_df] = args
    r = combine_rule(baserule,rule)
    r = evaluate_rule(r,outcome_df)
    return r

def get_best_rules(front, allrules,outcome_df,min_odds,criteria='info'):
    new_rules = []
    bests = {}
    best_thresholds = {}
    if len(front) < 1:
        front = [None]
    minsplit = max(5,int(outcome_df.shape[0]/10))
    for baserule in front:
        combined_rules = joblib.Parallel(n_jobs=4)(joblib.delayed(combine_and_eval_rule)((baserule,r,outcome_df)) for r in allrules)
        for combined_rule in combined_rules:
            if valid_rule(combined_rule,minsplit,min_odds):
                if (baserule is not None) and combined_rule[criteria] <= baserule.get(criteria,0):
                    continue
                rname = stringify_features(combined_rule['features'])
                if bests.get(rname,0) < combined_rule[criteria]:
                    #look at best info/odds ratio fro each set of organs
                    bests[rname] = combined_rule[criteria]
                    #svae thresholds as a tie-breaker
                    best_thresholds[rname] = stringify_thresholds(combined_rule['thresholds'])
                new_rules.append(combined_rule)
    new_rules = filter_rules(new_rules,bests,best_thresholds,criteria)
    return new_rules
    

def valid_rule(r,min_split_size=5,min_odds=0,min_info=.01):
    if r['odds_ratio'] < min_odds:
        return False
    if r.get('info',0) <= min_info:
        return False
    if min(r['upper_count'],r['lower_count']) < min_split_size:
        return False
    return True

def add_sd_dose_clusters(sddf, 
                         clusterer = None,
                         features=None,
                         reducer=None,
                         organ_subset=None,
                         normalize = True,
                         prefix='',
                         return_score=False,
                         n_clusters = 4,
                        ):
    if clusterer is None:
        clusterer = BayesianGaussianMixture(n_init=5,
                                            n_components=n_clusters, 
                                            covariance_type="full",
                                            random_state=100)
    if features is None:
        features=['V35','V40','V45','V50','V55','V60','V65']
    if reducer is None:
        reducer= None#PCA(len(organ_list),whiten=True)
    if organ_subset is None:
        organ_subset = Const.organ_list[:]
    organ_positions = [Const.organ_list.index(o) for o in organ_subset]
    vals = np.stack(sddf[features].apply(lambda x: np.stack([np.array([ii[i] for i in organ_positions]).astype(float) for ii in x]).ravel(),axis=1).values)
    if normalize:
        vals = (vals - vals.mean(axis=0))/(vals.std(axis=0) + .01)
    if reducer is not None:
        vals = reducer.fit_transform(vals)
    df = pd.DataFrame(vals,index = sddf.index)
    clusters = clusterer.fit_predict(vals)
    new_df = sddf.copy()
    cname= prefix+'dose_clusters'
    new_df[cname] = clusters
    new_df = reorder_clusters(new_df,
                              cname,
                              by='mean_dose',
                              organ_list=organ_subset#order by mean dose to clustered organs
                             )
    if return_score:
        score = clusterer.score_samples(vals).mean()
        return new_df, score
    return new_df

def reorder_clusters(df,cname,by='mean_dose',organ_list=None):
    df = df.copy()
    df2 = df.copy()
    severities = {}
    clusts = sorted(df[cname].unique())
    getmean = lambda d: d[by].astype(float).mean()
    if organ_list is not None and Utils.iterable(df[by].iloc[0]):
        keep_idx = [Const.organ_list.index(o) for o in organ_list]
        df[by] = df[by].apply(lambda x: [x[i] for i in keep_idx])
    if Utils.iterable(df[by].iloc[0]):
        getmean = lambda d: np.stack(d[by].apply(lambda x: np.array(x).sum()).values).mean()
    for c in clusts:
        subset = df[df[cname] == c]
        avg_severity = getmean(subset)
        severities[c] = avg_severity
    clust_order = np.argsort(sorted(severities.keys(), key = lambda x: severities[x]))
    clust_map = {c: clust_order[i] for i,c in enumerate(clusts)}
    df2[cname] = df[cname].apply(lambda x: clust_map.get(x))
    return df2


def get_aggregate_outcome(df,symptoms,dates,aggfunc=None):
    if aggfunc is None:
        aggfunc = np.max
    df = df.copy()
    date_idxs = [i for i,d in enumerate(df.dates.iloc[0]) if d in dates]
#     print('dates',dates,date_idxs)
    s_array = np.zeros((df.shape[0],len(symptoms)))
    for col,symptom in enumerate(symptoms):
        if 'symptoms_'+symptom not in df.columns:
            print('missing',symptom)
        svals = df['symptoms_'+symptom].apply(lambda x: np.max([x[i] for i in date_idxs]) )
        s_array[:,col] = svals
    res = np.apply_along_axis(aggfunc,1,s_array)
    return res

def var_tests(df, testcol, ycol,xcols, 
             regularize = True,
             scale=True):
    df = df.fillna(0)
    y = df[ycol]
    if y.max() > 1:
        y = y/y.max()
    if testcol not in xcols:
        xcols = xcols + [testcol]
    x = df[xcols].astype(float)
    if regularize:
        for col in xcols:
            x[col] = (x[col] - x[col].mean())/(x[col].std()+ .01)
    if scale:
        for col in xcols:
            x[col] = (x[col] - x[col].min())/(x[col].max() - x[col].min())
    for col in xcols:
        if x[col].std() < .00001:
#             print(col)
            x = x.drop(col,axis=1)
    x2 = x.copy()
    x2 = x2.drop(testcol,axis=1)
    boolean = (df[ycol].max() <= 1) and (len(df[ycol].unique()) <= 2)
    if boolean:
        model = sm.Logit
        method = 'bfgs'
        
    else:
        model = sm.OLS
        method= 'qr'
    logit = model(y,x)
    logit_res = logit.fit(maxiter=500,
                          disp=False,
                          method=method,
                         )
    
    logit2 = model(y,x2)
    logit2_res = logit2.fit(maxiter=500,
                            disp=False,
                            method=method,
                           )
    
    llr_stat = 2*(logit_res.llf - logit2_res.llf)
    llr_p_val = chi2.sf(llr_stat,1)
    
    aic_diff = logit_res.aic - logit2_res.aic
    bic_diff = logit_res.bic - logit2_res.bic
    odds = np.exp(logit_res.params)
    results = {
        'ttest_pval': logit_res.pvalues[testcol],
        'ttest_tval': logit_res.tvalues[testcol],
        'lrt_pval': llr_p_val,
        'aic_diff': aic_diff,
        'bic_diff': bic_diff,
        'odds_ratio': odds[testcol],
    }
    return results

def multi_var_tests(df, testcols, ycol,xcols, 
#              boolean=True,
             regularize = True,
             scale=True):
    df = df.fillna(0)
    y = df[ycol]
    if y.max() > 1:
        y = y/y.max()
    xcols = list(set(xcols).union(set(testcols)))
    x = df[xcols].astype(float)
    if regularize:
        for col in xcols:
            x[col] = (x[col] - x[col].mean())/(x[col].std()+ .01)
    if scale:
        for col in xcols:
            x[col] = (x[col] - x[col].min())/(x[col].max() - x[col].min())
    for col in xcols:
        if x[col].std() < .00001:
#             print(col)
            x = x.drop(col,axis=1)
    x2 = x.copy()
    x2 = x2.drop(testcols,axis=1)
    boolean = (df[ycol].max() <= 1) and (len(df[ycol].unique()) <= 2)
    if boolean:
        model = sm.Logit
        method = 'bfgs'
        
    else:
        model = sm.OLS
        method= 'qr'
    logit = model(y,x)
    logit_res = logit.fit(maxiter=500,
                          disp=False,
                          method=method,
                         )
    
    logit2 = model(y,x2)
    logit2_res = logit2.fit(maxiter=500,
                            disp=False,
                            method=method,
                           )
    
    llr_stat = 2*(logit_res.llf - logit2_res.llf)
    llr_p_val = chi2.sf(llr_stat,len(testcols))
    
    aic_diff = logit_res.aic - logit2_res.aic
    bic_diff = logit_res.bic - logit2_res.bic
    odds = np.exp(logit_res.params)
    results = {
        'lrt_pval': llr_p_val,
        'aic_diff': aic_diff,
        'bic_diff': bic_diff,
    }
#     for testcol in testcols:
#         results['odds_'+str(testcol)] = odds[testcol]
#         results['ttest_pval_' + str(testcol)]= logit_res.pvalues[testcol]
#         results['ttest_tval_' + str(testcol)]= logit_res.tvalues[testcol]
    return results

def get_stratification_metrics(y,ypred):
    #binary
    squeeze = lambda x: np.argmax(x,axis=1).ravel()
#     y_true = pd.get_dummies(y.loc[:,model.classes_]).values#one-hot encoe
    y_true = y.reshape(-1,1)#binary output shoud work like this idk
    roc = roc_auc_score(y_true,ypred[:,1])
    accuracy = accuracy_score(y_true, squeeze(ypred))
#     fscore = f1_score(y_true,squeeze(ypred))
    [precision,recall,fscore,support] = precision_recall_fscore_support(y_true,squeeze(ypred),average='binary')
    fbeta = lambda b: (1+b**2)*(precision*recall)/((b**2)*precision + recall)
    f_half = fbeta(.5)
    f2 = fbeta(2)
    matthews = matthews_corrcoef(y_true,squeeze(ypred))
    dor = (recall*precision)/((1-recall)*(1-precision))
    results=  {
        'roc': roc, 
        'mcc': matthews,
        'dor': dor,
        'accuracy': accuracy,
        'precision': precision,
        'recall':recall,
        'f1': fscore,
        'f_half': f_half,
        'f2': f2,
    }
    return results


class ClusterTester():
    
    outlier_ids = set([ ])
    def __init__(self,df,
                 cluster_organs, 
                 n_clusters=3, 
                 symptoms=None,
                 cluster_features=None,
                 outcome_dates=None,
                 default_confounders=None,
                 agg_type='max',
                 filter_outliers = False,
                 **kwargs,
                ):
        
        self.cluster_organs = cluster_organs
        self.n_clusters = n_clusters
        if symptoms is None:
            symptoms = ['drymouth']
        self.symptoms = symptoms
        
        if cluster_features is None:
            cluster_features = ['V35','V40','V45','V50','V55','V60']
        self.cluster_features = cluster_features
        
        if outcome_dates is None:
            outcome_dates = [13,33]
        self.outcome_dates = outcome_dates
        
        if agg_type == 'mean' or agg_type == 'average':
            self.agg_func = np.nanmean
        else:
            self.agg_func = np.nanmax
            
            
        if default_confounders is None:
            default_confounders = [
                't_severe',
                'n_severe',
                'performance_1',
                'performance_2',
                'hpv',
                'age_65',
                'BOT','Tonsil',
                'Parotid_Gland_limit',
#                 'IMRT','VMAT'
            ]
        self.default_confounders = default_confounders
        df = df.copy()
        df = self.filter_df(df,filter_outliers= filter_outliers)
        df = self.add_total_cluster_doses(df)
        self.df = df
        self.cluster_df=None
        self.current_mimic = None
        self.current_rules = None
        self.rule_candidates = None
        
    def add_total_cluster_doses(self,df,cols=None):
        df = df.copy()
        if cols is None:
            cols = ['mean_dose'] + self.cluster_features
        opositions = [Const.organ_list.index(o) for o in self.cluster_organs if o in Const.organ_list]
        for col in cols:
            if col in df.columns:
                df['total_'+col+'_cluster'] = df[col].apply(lambda x: np.sum([x[i] for i in opositions]))
        return df
    
    def toggle_use_change(self,value=None):
        if value == None:
            value = (not self.use_outcome_change)
        self.use_outcome_change = value
        
    def get_cluster_df(self,
                       resample=False,
                       use_mimic=False,
                       df = None,
                       use_cached=True,
                       **kwargs):
        # resampling is weird so I'd just avooid it and try to do it before passing df
        #if not caching and afterwords otherwise
        if df is None:
            df = self.df.copy()
        if resample:
            df = df.sample(frac=1)
        if use_mimic:
            if self.current_mimic is not None and use_cached and not (resample):
                df = self.current_mimic.copy()
            else:
                df, _ = self.get_mimic_clusters(df=df,max_rules=1,save_rule=(not resample),**kwargs)
            df = df.copy()
            df['dose_clusters'] = df['mimic_cluster_'+str(self.n_clusters-1)+'_0'].astype(int)
        else:
            if use_cached and self.cluster_df is not None and (not resample):
                df = self.cluster_df.copy()
            else:
                df,score = add_sd_dose_clusters(df,
                                    features=self.cluster_features,
                                    organ_subset=self.cluster_organs,
                                    return_score=True,
                                    n_clusters = self.n_clusters)
                self.score = score
        return df
    
    def get_outcome(self,threshold=5,df=None,use_change=False):
        if df is None:
            df = self.df.copy()
        outcome = get_aggregate_outcome(df,self.symptoms,self.outcome_dates,aggfunc=self.agg_func)
        if use_change:
            baseline = get_aggregate_outcome(df,self.symptoms,[0],aggfunc=self.agg_func)
            outcome=outcome-baseline
        if threshold > 0:
            outcome = outcome >= threshold
        return outcome
    
    def df_with_outcome(self,key='outcome',threshold=5):
        df = self.df.copy()
        df[key] = self.get_outcome(threshold=threshold)
        return df
        
    def filter_df(self,df,filter_outliers=False):
        keywords = ['_original','_max_','_6wk_symptoms','_late_symptoms']
        for keyword in keywords:
            to_drop = [col for col in df.columns if keyword in col]
            df = df.drop(to_drop,axis=1)
        if filter_outliers:
            iname = df.index.name
            df = df.reset_index()
            df = df[df.id.apply(lambda x: x not in ClusterTester.outlier_ids)]
            df = df.set_index(iname)
            print('after filter',df.shape)
        return df
    
    def extract_dose_vals(self,organ_list,df=None,features=None,as_df=True):
        oidxs = [Const.organ_list.index(o) for o in organ_list if o in Const.organ_list]
        
        if features is None:
            features = self.cluster_features
        if df is None:
            df = self.df.copy()
        vals = []
        names = []
        for f in features:
            for (oname, oidx) in zip(organ_list, oidxs):
                values = df[f].apply(lambda x: x[oidx])
                vals.append(values.values.reshape((-1,1)))
                names.append(f+'_'+oname)
        vals = np.hstack(vals)
        if as_df:
            vals = pd.DataFrame(vals,columns=names,index=self.df.index)
        return vals
    
    def extract_confounders(self,df=None,categorical_confounders=None,use_dose_confounders=False,organ_confounders=None,organ_confounder_features=None):
        if df is None:
            df = self.df.copy()
        if categorical_confounders is None:
            categorical_confounders = [dc for dc in self.default_confounders if dc != 'age']
            if 'age' in self.default_confounders:
                categorical_confounders = categorical_confounders + ['old']
        categorical_confounders = [c for c in categorical_confounders if c in df.columns]
        confounders = df[categorical_confounders].astype('float')
        if organ_confounders is None and use_dose_confounders:
            organ_confounders = [o for o in Const.organ_list if o not in self.cluster_organs]
        if organ_confounders is not None:
            if organ_confounder_features is None:
                organ_confounder_features = self.cluster_features
            dose_confounders = self.extract_dose_vals(
                organ_confounders,
                df = df,
                features=organ_confounder_features)
            confounders = pd.concat([dose_confounders,confounders],axis=1)
        return confounders
    
    def resample_df(self):
        df = self.df.copy().sample(frac=1)
    
    def propensity_scores(self,
                          thresholds=None,
                          use_dose_confounders=False,
                          confounder_pval_filter=1,
                          clusters=None,
                          use_mimic=False,
                         **kwargs):
        df = self.get_cluster_df(use_mimic=use_mimic)
        if clusters is None:
            clusters = list(df.dose_clusters.unique())
        #this doesn't actually affect propensity score but I have it because I'm dumb
        #will affect automatically determined confounders so 5 by default
        if thresholds is None:
            thresholds=[5]
        confounders = self.extract_confounders(
                df=df,
                use_dose_confounders=use_dose_confounders,
                **kwargs
        )
        
        results = []
        for c in clusters:
            treatment = df['dose_clusters'].apply(lambda x: x==c).values.ravel()
            for threshold in thresholds:
                outcome = self.get_outcome(threshold=threshold,df=df).ravel()
                if confounder_pval_filter < 1:
                    conf = filter_confounders(confounders,treatment,outcome,
                                                         max_pval=confounder_pval_filter)
                    confounder_names = list(conf.columns)
                    conf=conf.values
                else:
                    confounder_names = list(confounders.columns)
                    conf = confounders.values
                    
                entry = {
                    'cluster': c,
                    'threshold': threshold,
                    'confounders': confounder_names,
                }
                
                prop_df = propensity_df(treatment,outcome,conf)
                untreated = prop_df[prop_df.treatment.apply(lambda x: not x)].propensity.values
                treated = prop_df[prop_df.treatment].propensity.values
                for name, vals in zip(['untreated','treated'],[untreated,treated]):
                    entry[name] = vals
                    entry[name+'_mean'] = np.nanmean(vals)
                    qvals = [.05,.25,.5,.75,.95]
                    quantiles = np.quantile(vals,qvals)
                    for quant, val in zip(qvals,quantiles):
                        entry[name+'_'+str(quant)] = val
                    
                results.append(entry)
        return pd.DataFrame(results)
            
    def all_propensity_scores(self,**kwargs):
        base = self.propensity_scores(**kwargs)
        mimic = self.propensity_scores(**kwargs)
        mimic = mimic[mimic.cluster == 1]
        mimic.cluster = 'simple'
        return pd.concat([base,mimic],ignore_index=True)
    
    def get_ate(self,
            use_iptw=True,
            use_dr=True,
            use_matching=False,
            thresholds=[5,-5],
            use_dose_confounders=False,
            n_iters = 10,
            confounder_pval_filter=.1,
            skip_first = False,
            clusters=None,
            aggregate = True,
            use_mimic=False,
            **kwargs):
        resample_df = (n_iters > 1)
        arglist = []
        base_df = self.get_cluster_df(resample=False,use_mimic=use_mimic)
        for n in range(n_iters):
            if resample_df:
                df =self.get_cluster_df(resample=resample_df,use_cached=False,use_mimic=use_mimic)
            else:
                df = base_df.copy()
            if clusters is None:
                clustvals = df['dose_clusters'].unique()
            else:
                clustvals = clusters[:]
                skip_first = False
            confounders = self.extract_confounders(
                df= df,
                use_dose_confounders=use_dose_confounders,
                **kwargs)
            
            for c in clustvals:
                if c == np.min(clustvals) and skip_first:
                    continue
                treatment = df['dose_clusters'].apply(lambda x: x == c).values.ravel()
                for threshold in thresholds:
                    use_change= (threshold < 0)
                    t = np.abs(threshold)
                    outcome = self.get_outcome(threshold=t,use_change=use_change,df=df).ravel()
                    args = (treatment,outcome,confounders,threshold,use_iptw,use_dr,use_matching,confounder_pval_filter,c)
                    arglist.append(args)
        results = joblib.Parallel(n_jobs=4)(joblib.delayed(ate_worker)(args) for args in arglist)
        results = pd.DataFrame(results)
        results = results.sort_values(['cluster_value','threshold'],kind='mergesort')
        if aggregate and n_iters > 1:
            results = aggregate_ate_results(results)
        return results

    def get_basic_correlation(self,n_iters = 1, cluster=None, threshold = 0,use_mimic=False):
        pvals = []
        odds = []
        #gets pvalue or odds ratio with optional bootstrapping
        #will use fisher exact is both cluster and outcome are boolean
        #cluster = None and threshold = 0 use categorical
        if use_mimic:
            cluster=1
        use_chi2 = (cluster==None) or threshold < 1
        for n in range(n_iters):
            resample = n_iters > 1
            df = self.get_inference_df(threshold=threshold,cluster=cluster,use_mimic=use_mimic)
            x = df['x']
            outcome = df['outcome']
            try:
                if use_chi2:
                    odds_ratio,pval = vector_chi2(x,outcome)
                else:
                    odds_ratio, pval = boolean_fisher_exact(x,outcome)
                pvals.append(pval)
                odds.append(odds_ratio)
            except Exception as e:
                print(e)
        return pvals, odds
    
    def get_base_correlation_df(self,n_iters=1,thresholds=None,use_mimic=False):
        #get pvalue correlations for all/individual clusters either as a linear thing
        #or with different threhsolds
        results = []
        clusters = [None] + [i for i in range(self.n_clusters)]
        #threshold = 0 will be 1-10 isntead of the boolean thing
        if thresholds is None:
            thresholds = [0,3,5,7]
        for thold in thresholds:
            for clust in clusters:
                pvals, odds = self.get_basic_correlation(cluster=clust,
                                                         threshold=thold,
                                                         use_mimic=use_mimic,
                                                         n_iters=n_iters)
                for p,o in zip(pvals,odds):
                    entry = {
                        'threshold': thold,
                        'cluster': clust if clust is not None else -1,
                        'pval': p,
                        'effect_size': o,
                    }
                    results.append(entry)
        return pd.DataFrame(results)
    
    
    def get_inference_df(self,
                         cluster=None,
                         threshold=0,
                         resample=False,
                         use_mimic=False,
                         onehotify=False,
                         confounders=None,
                        use_change=False):
        #I wrote this after the ate stuff so it won't update in that funcion call if you change this
        df = self.get_cluster_df(resample=resample,use_mimic=use_mimic)
        df['x'] = df['dose_clusters']
        if cluster is not None and cluster >= 0:
            df['x'] = df['dose_clusters'].apply(lambda x: x==cluster)
        df['outcome'] = self.get_outcome(threshold=threshold,use_change=use_change)
        to_keep = ['x','outcome']
        if confounders is not None:
            to_keep.extend(confounders)
        to_keep = [c for c in to_keep if c in df.columns]
        df = df[to_keep]
        if onehotify:
            ignore = ['outcome']
            for c in confounders:
                if len(df[c].unique() < 5):
                    ignore = ignore + [c]
            if len(df['x'].unique()) <= 2:
                ignore = ignore + ['x']
            if len([c for c in df.columns if c not in ignore]) > 0:
                df = Utils.onehotify(df,ignore=ignore,drop_first=True)
        return df
    
    def get_lrt_correlations(self,confounders=None,
                             resample=False,
                             threshold=0,
                             use_mimic=False,
                             use_individual_effects=True,
                             cluster=None,
                            **kwargs):
        if confounders is None:
            confounders =self.default_confounders[:]
        df = self.get_inference_df(threshold=threshold,
                                   onehotify=True,
                                   cluster=cluster,
                                   use_mimic=use_mimic,
                                   resample=resample,
                                   confounders=confounders,
                                  **kwargs)
        treatment = ['x']
        #this wouldn't work if you have aconfounder with a similar label idk
        confounders = [c for c in df.columns if np.any([cc in c for cc in confounders])]
        if cluster is None and not use_mimic and self.n_clusters > 2:
            treatment = [c for c in df.columns if 'x_' in c and c not in confounders]
            results = multi_var_tests(df.fillna(0),treatment,'outcome',confounders)
        else:
            results = var_tests(df.fillna(0),treatment[0],'outcome',confounders)
        return results
    
    def get_lrt_correlation_df(self,
                               thresholds=None,
                               delta_thresholds = None,
                               confounder_list=None,
                               resample=False,
                               use_mimic=False,
                               include_delong=True,
                               clusters=None):
        #get pvalue correlations for all/individual clusters either as a linear thing
        #or with different threhsolds
        results = []
        #because it does individual coefficients you doen't really need other clusters idk
        if clusters is None:
            clusters = [None]
        #threshold = 0 will be 1-10 isntead of the boolean thing
        if thresholds is None:
            thresholds = [0,3,5,7] #-1 will be change
        if delta_thresholds is None:
            delta_thresholds = [0,3,5,7]
        if confounder_list is None:
            confounder_list = [self.default_confounders]
            
        def run_threshold(thold,use_change):
            for clust in clusters:
                for confounders in confounder_list:
                    res = self.get_lrt_correlations(cluster=clust,
                                                    threshold=thold,
                                                    resample=resample,
                                                    use_mimic=use_mimic,
                                                    use_change=use_change,
                                                    confounders=confounders)
                    
                    
                    entry = {
                        'cluster': clust if clust is not None else -1,
                        'confounders': confounders,
                        'outcome_change': use_change,
                    }
                    if use_change:
                        if thold == 0:
                            entry['threshold'] = -1
                        else:
                            entry['threshold'] = -thold
                    else:
                        entry['threshold'] = thold
                        
                    for k,v in res.items():
                        entry[k] = v
                    if include_delong:
                        pval = 1
                        if thold > 0:
                            pval = self.delong_roc(cluster=clust,threshold=thold,use_mimic=use_mimic,confounders=confounders,use_change=use_change)
                        dcol = 'delong_pval' 
                        entry[dcol] = pval
                    results.append(entry)
                    
        for thresh in thresholds:
            run_threshold(thresh,False)
        for dthresh in delta_thresholds:
            run_threshold(dthresh,True)
            
        return pd.DataFrame(results)
    
    def predict_cv(self,
                   confounders=None,
                   threshold=5,
                   cluster=None,
                   use_mimic=False,
                   resample=False,            
                   model=None,**kwargs):
        if confounders is None:
            confounders =self.default_confounders
        df = self.get_inference_df(threshold=threshold,
                                   onehotify=True,
                                   cluster=cluster,
                                   resample=resample,
                                   use_mimic=use_mimic,
                                   confounders=confounders,**kwargs)
        treatment = ['x']
        
        confounders = [c for c in df.columns if np.any([cc in c for cc in confounders])]
        if cluster is None and not use_mimic and self.n_clusters > 2:
            treatment = [c for c in df.columns if 'x_' in c and c not in confounders]
        xcols = treatment+confounders
        y = df['outcome'].values
        if model is None:
            model = LogisticRegression(penalty='elasticnet',l1_ratio=.5,solver='saga',class_weight='balanced',random_state=0)
        scale = lambda x: (x - x.min())/(x.max()-x.min())
        for c in treatment+confounders:
            if df[c].dtype != bool:
                df[c] = scale(df[c].fillna(0))
        ypred = predict_cv(model,df[treatment+confounders].values,y)
        ypred_baseline = predict_cv(model,df[confounders].values,y)
        return y, ypred, ypred_baseline
    
    def delong_roc(self,threshold=5,**kwargs):
        y, ypred, ypred_baseline = self.predict_cv(threshold=threshold,**kwargs)
        return delong(y.astype(int), ypred[:,1], ypred_baseline[:,1])[0][0]
        
    def get_cv_auc(self,model=None,**kwargs):
        if model is None:
            model = LogisticRegression(penalty='elasticnet',l1_ratio=.5,solver='saga',class_weight='balanced',random_state=0)
        y, ypred, ypred_baseline = self.predict_cv(model=model,**kwargs)
        metrics = get_stratification_metrics(y,ypred)
        metrics_baseline = get_stratification_metrics(y,ypred_baseline)
        
        results = {}
        for k,v in metrics.items():
            results[k] = v
            baseline = metrics_baseline[k]
            if baseline is None:
                baseline = 0
            results[k+'_change'] = v - baseline
        return results
    
    def get_demographic_breakdown(self,
                                  cat_cols = None, 
                                  cont_cols = None,
                                  include_outcome=True,
                                  use_mimic=False,
                                 ):
        df = self.get_cluster_df(use_mimic=use_mimic)
        df.n_stage = df.n_stage.apply(lambda x: 'n2b' if x == 'n2' else x)
        clust_col = 'dose_clusters'
        if cat_cols is None:
            cat_cols = [
                'is_male','t_stage','n_stage','hpv',
                'performance_score',
                'ic','os','subsite','BOT','Tonsil',
                'previously_treated','ic_prior_to_enrollment',
                'rt_prior_to_enrollment','concurrent_prior_to_enrollment',
                'sx_prior_to_enrollment','sx_prior_to_enrollment',
                'Technique',
            ] + [c for c in df.columns if '_limit' in c]
        if cont_cols is None:
            cont_cols = [
                'age',
                'followup_days',
            ] + [c for c in df.columns if 'total_' in c]
        if include_outcome:
            outcome_name = '-'.join(self.symptoms)
            for thold in [3,5,7]:
                cname = outcome_name+'>'+str(thold)
                df[cname] = self.get_outcome(threshold=thold)
                cat_cols.append(cname)
        results = {'counts': pd.Series({c: d.shape[0] for c,d in df.groupby('dose_clusters')},name='cluster')}
        for col in cat_cols:
            if np.issubdtype(df[col].dtype,np.number):
                catdf = df.fillna(-1,downcast='infer')
            else:
                catdf = df.fillna('unknown').astype(str)
#             catdf = df[~df[col].isnull()]
            entrys = {}
            vchi2 = vector_chi2(catdf[clust_col],catdf[col])
            for clust, subdf in catdf.groupby(clust_col):
                subchi2 = vector_chi2(catdf.dose_clusters.apply(lambda x: x == clust), catdf[col])
                clust_entry = {'pval': subchi2[1],'tval':subchi2[0],'overall_pval':vchi2[1]}
                csize = subdf.shape[0]
                for vName, subsubdf in subdf.groupby(col):
                    count = subsubdf.shape[0]
                    clust_entry[vName] = str(count) + ' (' + str(np.round(count*100/csize,1)) + ')%'
                entrys[clust] = clust_entry
            edf = pd.DataFrame(entrys).T
            edf.index.name = 'cluster'
            results[col] = edf
        cont_results = {}
        def get_dist(vals):
            vals = vals.astype(float)
            [q5,median,q95] = vals.quantile([.05,.5,.95])
            mean = vals.mean()
            std = vals.std()
            ce = {
                'mean': mean,
                'std': std,
                'q5': q5,
                'median': median,
                'q95': q95,
            }
            return ce
        
        for col in cont_cols:
            tempdf = df[~df[col].isnull()]
            
            combined_entry = get_dist(tempdf[col])
            tempchi2 = vector_chi2(tempdf.dose_clusters, tempdf[col])
            combined_entry['tval'] = tempchi2[0]
            combined_entry['pval'] = tempchi2[1]
            entrys = {'total': combined_entry}
            for clust, subdf in tempdf.groupby(clust_col):
                clust_entry = get_dist(subdf[col])
                tempchi2 = vector_chi2(tempdf.dose_clusters.apply(lambda x: x == clust), tempdf[col])
                clust_entry['tval'] = tempchi2[0]
                clust_entry['pval'] = tempchi2[1]
                entrys[clust] = clust_entry
            edf = pd.DataFrame(entrys).T
            edf.index.name = 'cluster'
            results[col] = edf
        return results
    
    def get_current_rules(self,**kwargs):
        if self.current_rules is None:
            self.get_mimic_clusters(**kwargs)
        rule = self.current_rules[self.n_clusters-1]
        return rule[0]
    
    def show_all_rules(self,**kwargs):
        string = ''
        if self.rule_candidates is None:
            self.get_mimic_clusters(**kwargs)
        for rule in self.rule_candidates:
            string += self.show_rule(rule=rule)
            string += ' \r\n '
            string += '_________________'
            string += ' \r\n '
        return string
    
    def show_rule(self,rule=None,**kwargs):
        if rule is None:
            rule = self.get_current_rules(**kwargs)
        f = rule.get('features')
        t = rule.get('thresholds')
        string = ''
        if f is not None and t is not None:
            for ff,tt in zip(f,t):
                string += str(ff) + '>' + str(tt) + ', '
        string = string[:len(string)-2] + ' \r\n'
        metrics = [k for k in rule.keys() if 'roc_' in k or 'cluster_' in k]
        for m in metrics:
            string += ' ' + m + ':' + str(np.round(rule.get(m,-1),3))
        return string[:len(string)-2]
    
    def get_mimic_clusters(self,
                           df=None,
                           granularity=1,
                           max_frontier=20,
                           max_rules=6,
                           criteria='info',
                           useLimits=False,
                           target_clusters=None,
                           min_odds = 0,
                           min_info= .08,
                           save_rule=None,
                           maxdepth=4):
        if df is None:
            df = self.get_cluster_df()
            if save_rule is None:
                save_rule = True
        elif save_rule is None:
            save_rule=False
            
        organs =self.cluster_organs[:]
#         features = self.cluster_features[:] + ['mean_dose','max_dose']
        features = get_all_dvh(self) + ['mean_dose','max_dose']
        outcome = self.get_outcome(df=df,threshold=-1)
        outcome_diff = self.get_outcome(df=df,use_change=True,threshold=-1)
        dose_df = extract_dose_vals(df,organs,features,include_limits=useLimits)

        if target_clusters == None:
            target_clusters = [self.n_clusters -1]
        rulesets = {}
        for target_cluster in target_clusters:
            df['y'] = df.dose_clusters.apply(lambda x: x == target_cluster)
            y = df[['y']]
            rules = get_rule_df(dose_df,y,min_odds=min_odds,granularity=granularity)
            
            sort_rules = lambda rlist: sorted(rlist, key=lambda x: -x[criteria])
            rules = sort_rules(rules)
            min_info = min(rules[0].get('info',0.0)*.6,float(min_info))
            rules = [r for r in rules if r.get('info',0) >= min_info]
#             print('n rules',len(rules))
            frontier = [None]
            #idk sometime it does'nt work
            best_rules = [rules[0]]
            depth = 0
            while (depth < maxdepth) and (frontier is not None) and (len(frontier) > 0):
                frontier = get_best_rules(frontier,rules,y,min_odds=min_odds,criteria=criteria)
                frontier = sorted(frontier, key = lambda x: -x[criteria] if x is not None else 0)
                frontier = frontier[:max_frontier]
                depth += 1
                best_rules.extend(frontier)
            best_rules = sort_rules(best_rules)
            best_rules = best_rules[:max_rules]
            best_rules = [self.eval_rule(r,y,outcome,outcome_diff) for r in best_rules]
            for i,rule in enumerate(best_rules):
                df['mimic_cluster_'+str(target_cluster)+'_'+str(i)] = rule['rule'].astype(int).values
            rulesets[target_cluster]= best_rules
        if save_rule:
            self.current_mimic = df.copy()
            self.rule_candidates = best_rules[:]
            self.current_rules = {k:v for k,v in rulesets.items()}
        return df, rulesets
    
    def eval_rule(self,r,y,outcome,outcome_diff):
        rule = r['rule'].values.reshape(-1,1)
        for threshold in [3,5,7]:
            o = (outcome > threshold)
            ot = (outcome_diff > threshold)
            
            r['roc_cluster'] = roc_auc_score(y,rule)
            r['roc_outcome_' + str(threshold)] = roc_auc_score(o,rule)
            r['roc_outcome_change_' + str(threshold)] = roc_auc_score(ot,rule)
            
            yprecision, yrecall, yf1, ysupport = precision_recall_fscore_support(y.values.ravel(),rule.ravel())
            r['cluster_precision'] = yprecision
            r['cluster_recall'] = yrecall
            r['cluster_f1'] = yf1
        return r
    
    def unsupervised_predictor(self,cluster=None,use_mimic=False,**kwargs):
        #converts cluster into a prediction to user for stuff
        df = self.get_cluster_df(use_mimic=use_mimic,**kwargs)
        clusters = df.dose_clusters.values
        if cluster is not None and cluster >= 0:
            clusters = (clusters == cluster).astype(int)
        else:
            clusters = clusters/clusters.max()
        #get_stratificaiotn metrics assumes the output is 2 dimensional    
        clusters = clusters.reshape(-1,1)
        ypred = np.concatenate([1-clusters,clusters],axis=1)
        return ypred 
    
    def unsupervised_metrics(self,use_mimic=False,threshold=5,model=None,**kwargs):
        #gives model metrics using just the clusters as the prediction values
        ypred = self.unsupervised_predictor(use_mimic=use_mimic,**kwargs)
        outcome = self.get_outcome(threshold=threshold)
        if model is None:
            model = LogisticRegression(penalty='elasticnet',l1_ratio=.5,solver='saga',class_weight='balanced',random_state=0)
        return get_stratification_metrics(outcome,ypred)