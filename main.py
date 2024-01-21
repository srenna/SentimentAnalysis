from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# import the data
df = pd.read_csv('reviews.csv', delimiter='\t')

# load the train.csv
train_df = pd.read_csv('train.csv')

# load the valid.csv
valid_df = pd.read_csv('valid.csv')

# check the data
# df.head()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join the tokens back into text
    text = ' '.join(tokens)

    return text


def categorize_rating(rating):
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive


df['Sentiment'] = df['RatingValue'].apply(categorize_rating)

# count how many positive, neutral and negative reviews there are
# df['Sentiment'].value_counts()

# min_count = df['Sentiment'].value_counts().min()
average_count = df['Sentiment'].value_counts().iloc[1:].mean().astype(int)

# Create a list to store the downsampled DataFrames
downsampled_dfs = []

# Iterate over each sentiment
for sentiment in [2, 1, 0]:
    sentiment_df = df[df['Sentiment'] == sentiment]
    if sentiment == 2:
        # For positive sentiment, randomly sample to match the minimum count
        downsampled_df = sentiment_df.sample(n=average_count, random_state=42)
    else:
        # For negative and neutral sentiments, use all available data
        downsampled_df = sentiment_df
    downsampled_dfs.append(downsampled_df)

# Concatenate the downsampled DataFrames to create a balanced dataset
balanced_df = pd.concat(downsampled_dfs)

# print the count of each sentiment class to verify
# print(balanced_df['Sentiment'].value_counts())

# from balanced_df, create a dataframe with only the Sentiment and Review
# columns
df = balanced_df[['Sentiment', 'Review']].reset_index(drop=True)

# Split the data into training and validation sets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=243)

# Save the training and validation sets to CSV files
train_df.to_csv('train.csv', index=False)
valid_df.to_csv('valid.csv', index=False)

X_train = train_df['Review']
y_train = train_df['Sentiment']

# preprocess the training and validation sets
X_train = X_train.apply(preprocess_text)

# Create a bag of words vectorizer
# with unigrams, bigrams, and trigrams
vectorizer = CountVectorizer(ngram_range=(1, 2))

X_train = vectorizer.fit_transform(X_train)

# Define the parameter grid for alpha values
param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}

# Train a Naive Bayes classifier
clf = MultinomialNB().fit(X_train, y_train)

# GridSearchCV
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_['alpha']

# Train the final model with the best alpha on the full training set
final_model = MultinomialNB(alpha=best_alpha)
final_model.fit(X_train, y_train)

# preprocess the text
X_valid_pre = valid_df['Review'].apply(preprocess_text)

# Make predictions on the validation data
X_valid = vectorizer.transform(X_valid_pre)
y_valid = valid_df['Sentiment']
y_pred = final_model.predict(X_valid)

# Calculate accuracy and print the classification report
accuracy_test = accuracy_score(y_valid, y_pred)

print(
    f"Accuracy: The accuracy on the test set is {accuracy_test * 100:.2f}%\n")

# classwise F1 scores
class_labels = ["Negative", "Neutral", "Positive"]

report_valid = classification_report(
    y_valid,
    y_pred,
    target_names=class_labels,
    output_dict=True)

f1_score = report_valid['macro avg']['f1-score']
print(f"Average F1 score: {f1_score:.2f}")
print()

print("Class-wise F1 scores:")
for class_name in class_labels:
    f1_score = report_valid[class_name]['f1-score']
    print(f"   {class_name}: {f1_score:.2f}")

# calculate the confusion matrix
cm_val = confusion_matrix(y_valid, y_pred, normalize='true')

print("\nConfusion Matrix:")
print(f"{' ':<12}{'Negative':<10}{'Neutral':<10}{'Positive':<10}")
for i, row in enumerate(cm_val):
    print(f"{class_labels[i]:<12}", end="")
    for val in row:
        print(f"{val:.3f}{' ':<6}", end="")
    print()
