import pandas as pd
def preprocess(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df