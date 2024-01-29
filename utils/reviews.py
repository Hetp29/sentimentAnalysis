import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.data import Field, TabularDataset, BucketIterator

def preprocessData(csv_path, test_size = 0.2, batch_size = 64):
    df = pd.read_csv(csv_path)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    train_df, test_df, = train_test_split(df, test_size = test_size, random_state = 42)