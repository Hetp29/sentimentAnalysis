import nntplib
import torch 
import torch.optim as optim 
import torch.nn.functional as F
import torch.nn as nn
from models.classifier import Classifier
from utils.reviews import preprocessData

Text, train_iterator, test_iterator = preprocessData('dataset/reviews.csv')

vocab_size = len(Text.vocab)
embedding_dim = 100
hidden_dim = 250
output_dim = 2

model = Classifier(vocab_size, embedding_dim, hidden_dim, output_dim)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

for epoch in range(5):
    model.train()
    for batch in train_iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.sentiment.float())
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_iterator:
        predictions = model(batch.text).squeeze(1) 
        rounded_preds = torch.round(torch.sigmoid(predictions))
        correct += (rounded_preds == batch.sentiment).sum().item()
        total += len(batch.sentiment)

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')

