import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data (replace with your dataset file path)
file_path = 'subset_data.csv'  # Replace with your dataset file path
subset_data = pd.read_csv(file_path)

# Preprocess the data
subset_data['combined_text'] = subset_data['Summary'].fillna('') + " " + subset_data['Text'].fillna('')

# Cache the TF-IDF vectorization and similarity computation
@st.cache_data
def compute_tfidf_and_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(data['combined_text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, cosine_sim

tfidf, cosine_sim = compute_tfidf_and_similarity(subset_data)

# Function to recommend similar products
def recommend_similar_products(product_id, top_n=5):
    if product_id not in subset_data['ProductId'].values:
        return ["Product not found!"]
    
    product_index = subset_data[subset_data['ProductId'] == product_id].index[0]
    similarity_scores = cosine_sim[product_index]
    similar_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]
    similar_products = subset_data['ProductId'].iloc[similar_indices].tolist()
    return similar_products

# Streamlit UI
st.title("TF-IDF Product Recommendation System")
st.write("Enter a product ID to get recommendations:")

# Input box for product ID
product_id_input = st.text_input("Product ID:")

# Number of recommendations
top_n = st.slider("Number of recommendations:", 1, 10, 5)

if product_id_input:
    st.write(f"Recommendations for Product ID {product_id_input}:")
    recommendations = recommend_similar_products(product_id_input, top_n=top_n)
    st.write(recommendations)