# SentimentAnalysis_BERT

## Problem statement
The goal is to train a deep neural network to predict the sentiment of (hate) speech text.

## Solution
This problem was solved by training a recent text classification model called BERT (Bidirectional Encoder Representations from Transformers).

The pretrained model was loaded from Hugging Faces' transformers. Then a two layer bidirectional nn.GRU sub-networks was appended at BERT's out. When input tokenized ids of text are passed through BERT then it outputs a representation (also called embedding in this case). This representation is then passed through GRU layers. Following the time-step from GRU the last time-step's hidden state is extracted and then passed through a logistic classifier (using nn.Linear module). Following this we train the model using nn.BCEWithLogitsLoss loss.

We used IMDB dataset instead of using some specific hate speech dataset. Surprisingly the model still nicely classifies the hate speech.
