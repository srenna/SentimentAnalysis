## Overview
The project focuses on building a sentiment classifier for categorizing restaurant reviews. Utilizing a dataset stored in reviews.csv with nearly 2000 reviews, the objective is to create a balanced sentiment classification model.

## Purpose
The primary goal is to train and test a Bag-of-Words (BoW) text classifier on review texts, using the RatingValue as a label. Ratings are binned into negative, neutral, and positive sentiments, labeled as 0, 1, and 2, respectively. The resulting data is processed and split into training and validation sets, saved as train.csv and valid.csv.

## Deliverables
The project includes a Python script (main.py) that loads, preprocesses, and splits the data. It trains the model using the training set and evaluates performance on the validation set. Metrics such as accuracy, F1-score, and a confusion matrix are printed.
