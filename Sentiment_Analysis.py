from transformers import pipeline
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
form sklearn.model_selection import train_test_split

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
    "This is such a great course!",
    "This is not good for me!",
    "I can't say this was a bad movie.",
    "I can't say this was a good movie."
])
