## Overview
The project focuses on building a sentiment classifier for categorizing restaurant reviews. Utilizing a dataset stored in reviews.csv with nearly 2000 reviews, the objective is to create a balanced sentiment classification model.

## Purpose
The primary goal is to train and test a Bag-of-Words (BoW) text classifier on review texts, using the RatingValue as a label. Ratings are binned into negative, neutral, and positive sentiments, labeled as 0, 1, and 2, respectively. 

## Web Scraping 
To gather reviews, a script is provided to scrape reviews from your website choice. Trustpilot is used as a sample. The provided Python script utilizes the requests module to download HTML from the specified Trustpilot URL. Employing BeautifulSoup, the code navigates through the review pages, extracts relevant information, and compiles it into a CSV file with columns including companyName, datePublished, ratingValue, and reviewBody. The process involves careful handling of HTML elements and limiting the number of reviews to 500. The final PY file runs as a script, saving the CSV file to the present working directory, and allowing seamless execution from a terminal or Jupyter environment.

## Deliverables
The project includes a Python script (main.py) that loads, preprocesses, and splits the data. It trains the model using the training set and evaluates performance on the validation set. Metrics such as accuracy, F1-score, and a confusion matrix are printed.
