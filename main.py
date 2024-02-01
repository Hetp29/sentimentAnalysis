import nntplib
import torch 
import torch.optim as optim 
import torch.nn.functional as F
from models.classifier import Classifier
from utils.reviews import preprocessData

Text, train_iterator, test_iterator = preprocessData('dataset/reviews.csv')

vocab_size = len(Text.vocab)
embedding_dim = 100
hidden_dim = 250
output_dim = 2

model = Classifier(vocab_size, embedding_dim, hidden_dim, output_dim)

optimizer = optim.Adam(model.parameters())
criterion = nntplib.BCEWithLogitsLoss()

