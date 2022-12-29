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



