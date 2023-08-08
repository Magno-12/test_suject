import pandas as pd
import numpy as np

def clean_signals(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return (df - df.min()) / (df.max() - df.min())
