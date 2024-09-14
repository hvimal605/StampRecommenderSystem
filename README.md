# Stamp Recommender System

The **Stamp Recommender System** allows users to search and receive recommendations for stamps or products based on their descriptions. The system uses **Natural Language Processing (NLP)** techniques, such as **TF-IDF Vectorization** and **Cosine Similarity**, to find and rank the most relevant results.

## Features

- **Search Functionality**  
  Users can input a description of a stamp or product to find matching entries from the dataset.

- **TF-IDF Vectorization**  
  The description data is processed using **TF-IDF** (Term Frequency-Inverse Document Frequency) to determine the importance of words in each entry.

- **Cosine Similarity**  
  This technique is used to calculate the similarity between the search query and existing stamp descriptions, ranking the most relevant matches.

- **Top 10 Recommendations**  
  Displays the top 10 stamps/products based on similarity to the user's input.

- **Interactive Interface**  
  A smooth and responsive interface built with **Streamlit** to deliver an enhanced user experience.

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (Pandas, NumPy, NLTK)
- **Data Processing**: Scikit-learn (TF-IDF Vectorizer, Cosine Similarity)
- **NLP**: NLTK (Natural Language Toolkit)
