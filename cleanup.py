import pandas as pd

def clean_signals(df):
    # Normaliza las señales
    return (df - df.min()) / (df.max() - df.min())
