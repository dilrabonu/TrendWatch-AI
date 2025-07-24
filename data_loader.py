import pandas as pd

def load_data(file):
    if file:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("data/cleaned_google_stock.csv")  

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df = df.dropna()
    return df
