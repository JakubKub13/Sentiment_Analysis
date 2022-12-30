from transformers import pipeline
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
# To get GPU device 
import torch

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Basic usage of pipeline
classifier = pipeline("sentiment-analysis")

# Check for type of classifier
print(type(classifier))

# Use classifier to predict sentiment
classifier("This is such a great course!")
classifier("This is not good for me!")
classifier("I can't say this was a bad movie.")
classifier("I can't say this was a good movie.")

# Multiple inputs passed as a list
classifier([
    "This is such a great movie!",
    "I can't understand any of this. Instructor kept telling me to meet the prequisites. What are prequisites? Why does he keep saying that?" 
])

torch.cuda.is_available() # Check if GPU is available
torch.cuda.current_device() # Check which GPU is being used

# Use the GPU with new pipleline object
classifier = pipeline("sentiment-analysis", device=0) # 0 is the GPU device
df = pd.read_csv('AirlineTweet.csv') # Read our csv file contianing the data
df.head() # Check what is inside our dataframe
# Filter out our dataframe to only contain the columns we need
df = df_[['airline_sentiment', 'text']].copy()
# Plot histogram of our data
df['airline_sentiment'].hist() # Tell us if data is balanced or not and what classes we have in our data

# Filter out our dataframe to only contain the columns we need in our case neutral
df = df[df.airline_sentiment != 'neutral'].copy() # our model can not predict neutral
# labels to integers
target_map = {'positive': 1, 'negative': 0}
df['target'] = df['airline_sentiment'].map(target_map)
# call .head funtion to check if our data is correct
df.head()
# check how many samples we have in our data
len(df) # check how much time this took for model to make prediction for each sample

# Now we use pretraided classifier to make predictions
texts = df['text'].tolist() # convert our text column to list
predictions = classifier(texts) # check how long this process takes

#print(predictions)
print(predictions) # check what is inside our predictions

# using AUC metric because our data are not balanced
# AUC is a good metric for imbalanced data
probs = [d['score'] if d['label'].startswith('P') else 1 - d['score'] for d in predictions]

print(probs)

# convert predictions to array
preds = [1 if d['label'].startswith('P') else 0 for d in predictions]

preds = np.array(preds)

# compute the accuracy
print("acc: ", np.mean(sf['target'] == preds))

# compute the confusion matrix
#  It is a summary of the model's predictions for a binary or multi-class classification problem,
cm = confusion_matrix(df['target'], preds, normalize='true') # normalize='true' each row will sum to 1
cm

# plot the confusion matrix
def plot_cm(cm):
    classes = ['negative', 'positive']
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    ax = sn.heatmap(df_cm, annot=True, fmt='g')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Target')

plot_cm(cm)

# check the f1 score
f1_score(df['target'], preds) 
# we invert targets and predictions
f1_score(1 - ds['target'], 1 - preds)
# Check auc score
roc_auc_score(df['target'], probs)
roc_auc_score(1 - df['target'], 1 - np.array(probs)) # demonstrate that auc is not sensitive to the order of the inputs



