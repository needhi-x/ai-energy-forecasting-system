def create_features(df):
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['lag1'] = df['energy'].shift(1)
    return df.dropna()