# =====================================
# TF-IDF
# =====================================


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define stop words
stop_words = set(stopwords.words('italian'))

# Preprocess text function
def preprocess_text(text):
    text = str(text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Path to the CSV file
fp_only = os.path.join(os.getcwd(), "ARTICLES DATA", "MERGED DATASET", "merged_output_FIRST_PAGE_ONLY.csv")

# Load the DataFrame
df_fp_only = pd.read_csv(fp_only)

# Directory to save TF-IDF dataframes
output_path_testata = os.path.join(os.getcwd(), "ARTICLES DATA", "MERGED DATASET", "TF-IDF DATAFRAME")
output_path_date = os.path.join(os.getcwd(), "ARTICLES DATA", "MERGED DATASET", "TF-IDF DATAFRAME", "DATE")

# Ensure the directories exist
os.makedirs(output_path_testata, exist_ok=True)
os.makedirs(output_path_date, exist_ok=True)

# Initialize dataframe to hold top words for testata and date
top_words_testata_df = pd.DataFrame()
top_words_date_df = pd.DataFrame()

# Process by testata
for testata in df_fp_only["testata"].unique():
    documents = df_fp_only["Text_lemm"][df_fp_only["testata"] == testata]
    documents = documents.apply(preprocess_text)
    print(f"Preprocessing done for {testata}")

    tfidf_matrix = vectorizer.fit_transform(documents)
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    df_tfidf.to_csv(os.path.join(output_path_testata, f"TEXT_LEMM_{testata}.csv"), index=False)
    print(f"Vectorization done for {testata}, shape: {df_tfidf.shape}")

    word_scores = df_tfidf.sum(axis=0)
    sorted_words = word_scores.sort_values(ascending=False)
    top_words = sorted_words.head(1000)

    top_words_testata_df[testata] = top_words.index[:1000]
    print(f"Top words for {testata} added to top_words_testata_df")

# Save the top words dataframe for testata
top_words_testata_df.to_csv(os.path.join(output_path_testata, "TEXT_LEMM_top_words.csv"), index=False)
print("Top words dataframe for testata saved.")

# Process by date
for date in df_fp_only["published_date"].unique():
    documents = df_fp_only["Text_lemm"][df_fp_only["published_date"] == date]
    documents = documents.apply(preprocess_text)
    print(f"Preprocessing done for {date}")

    tfidf_matrix = vectorizer.fit_transform(documents)
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    df_tfidf.to_csv(os.path.join(output_path_date, f"TEXT_LEMM_{date}.csv"), index=False)
    print(f"Vectorization done for {date}, shape: {df_tfidf.shape}")

    word_scores = df_tfidf.sum(axis=0)
    sorted_words = word_scores.sort_values(ascending=False)
    top_words = sorted_words.head(1000)

    top_words_date_df[date] = top_words.index[:1000]
    print(f"Top words for {date} added to top_words_date_df")

# Save the top words dataframe for date
top_words_date_df.to_csv(os.path.join(output_path_date, "TEXT_LEMM_top_words.csv"), index=False)
print("Top words dataframe for date saved.")
