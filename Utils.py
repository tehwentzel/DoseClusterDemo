import simplejson
import numpy as np
from Constants import Const
import datetime
import pandas as pd

def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True
    
def process_arrays(val):
    if isinstance(val,str):
        try: 
            return json.loads(val)
        except:
            return val
        
def onehotify(df,ignore=None,drop_first=False):
    df = df.copy()
    if ignore is None:
        ignore = set([])
    subdf = pd.concat([pd.get_dummies(df[c],prefix=c,drop_first=drop_first) for c in df.columns if c not in ignore],axis=1)
    if ignore is not None:
        return pd.concat([subdf,df[ignore]],axis=1)
    return subdf