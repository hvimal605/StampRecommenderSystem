import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Download the required NLTK data
nltk.download('punkt')

# Load the dataset with caching to improve performance
@st.cache_data
def load_data():
    # Ensure the correct path to the s21.csv file
    data = pd.read_csv('StampDataset.csv')
    return data

# Define tokenizer and stemmer
stemmer = SnowballStemmer('english')
def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Create stemmed tokens column
def create_stemmed_tokens_column(data):
    data['stemmed_tokens'] = data.apply(
        lambda row: tokenize_and_stem(str(row['Description'])), axis=1
    )
    return data

# Define TF-IDF vectorizer and cosine similarity function
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, token_pattern=None)
def cosine_sim(text1, text2):
    text1_concatenated = ' '.join(text1)
    text2_concatenated = ' '.join(text2)
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
    return cosine_similarity(tfidf_matrix)[0][1]

# Define search function
def search_entries(query, data):
    query_stemmed = tokenize_and_stem(query)
    data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
    results = data.sort_values(by=['similarity'], ascending=False).head(10)
    if results.empty:
        return None
    return results[['Id', 'Issue Date', 'Description', 'Denomination', 'similarity']]

# Main function to run the app
def main():
    st.title("Stamp/Product Search and Recommendation System")

    # Load and preprocess the data
    data = load_data()
    data = create_stemmed_tokens_column(data)

    # User input and search functionality
    query = st.text_input("Enter Description")
    submit = st.button('Search')

    if submit:
        if query:
            res = search_entries(query, data)
            if res is not None:
                st.write(res)
            else:
                st.write("No matching entries found. Please try a different query.")
        else:
            st.write("Please enter a description to search.")

if __name__ == "__main__":
    main()
