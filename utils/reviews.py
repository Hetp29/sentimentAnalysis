import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.data import Field, TabularDataset, BucketIterator

def preprocessData(csv_path, test_size = 0.2, batch_size = 64):
    df = pd.read_csv(csv_path)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    train_df, test_df = train_test_split(df, test_size = test_size, random_state = 42)
    
    Text = Field(sequential=True, tokenize='spacy', lower=True)
    Label = Field(sequential=False, use_vocab=False)
    
    fields = [('text', Text), ('sentiment', Label)]
    
    train_data, test_data = TabularDataset.splits(
        path = 'dataset/',
        train='train.csv',
        test='test.csv',
        format='csv',
        fields=fields
    )
    
    Text.build_vocab(train_data, max_size=25000) 
    train_iterator, test_iterator, = BucketIterator.splits(
        (train_data, test_data),
        batch_size = batch_size,
        sort_key = lambda x: len(x.text),
        repeat=False,
        shuffle=True
    )
    
    return Text, train_iterator, test_iterator

#reads csv files, converts sentiments to number values
#splits dataset into training and testing sets