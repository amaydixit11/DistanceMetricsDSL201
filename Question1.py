import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the folder path where the dataset is stored
folder_path = './Q1_dataset'

def create_term_document_matrix(folder_path):
    documents = []

    # Load documents from the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())  # Append the content of each document to the list

    # Tokenize and clean documents
    def clean_tokenize(doc):
        doc = re.sub(r'[^\w\s]', '', doc.lower())  # Remove punctuation and convert to lowercase
        tokens = doc.split()  # Split the cleaned document into words
        return tokens

    # Apply cleaning and tokenization to all documents
    tokenized_documents = [clean_tokenize(doc) for doc in documents]

    # Create vocabulary from tokenized documents
    vocabulary = set()
    for doc in tokenized_documents:
        vocabulary.update(doc)  # Add unique terms to the vocabulary set
    vocabulary = sorted(vocabulary)  # Sort to ensure consistent column ordering

    # Initialize the term-document matrix with zeros
    term_document_matrix = np.zeros((len(tokenized_documents), len(vocabulary)), dtype=int)

    # Create a mapping of terms to their corresponding index in the matrix
    term_to_index = {term: i for i, term in enumerate(vocabulary)}
    
    # Fill the term-document matrix with term counts
    for i, doc in enumerate(tokenized_documents):
        for term in doc:
            if term in term_to_index:
                term_index = term_to_index[term]  # Get the index of the term
                term_document_matrix[i, term_index] += 1  # Increment the count for the term

    # Create a DataFrame for better readability and return results
    df = pd.DataFrame(term_document_matrix, columns=vocabulary)
    
    return df, term_document_matrix, vocabulary

def jaccard_distance(doc1, doc2):
    # Compute Jaccard distance between two documents
    set1 = set(np.where(doc1 > 0)[0])  # Get indices of non-zero entries in doc1
    set2 = set(np.where(doc2 > 0)[0])  # Get indices of non-zero entries in doc2
    
    intersection = set1 & set2  # Calculate intersection of terms
    union = set1 | set2  # Calculate union of terms
    if len(union) == 0:
        return 0  # Return zero if both sets are empty
    j_coeff = len(intersection) / len(union)  # Jaccard similarity
    return 1 - j_coeff  # Jaccard distance

def euclidean_distance(doc1, doc2):
    # Calculate Euclidean distance between two document vectors
    return np.sqrt(np.sum((doc1 - doc2) ** 2))  # Return the Euclidean distance

def cosine_similarity(doc1, doc2):
    # Compute Cosine similarity between two documents
    dot_product = np.dot(doc1, doc2)  # Dot product of the two vectors
    norm1 = np.linalg.norm(doc1)  # Norm of the first vector
    norm2 = np.linalg.norm(doc2)  # Norm of the second vector
    if norm1 == 0 or norm2 == 0:
        return 0  # Return zero if either vector is zero
    return (dot_product / (norm1 * norm2))  # Return Cosine similarity

def KL_divergence(doc1, doc2):
    # Function to calculate Kullback-Leibler divergence between two documents
    def normalize(doc):
        total_terms = np.sum(doc)  # Sum of all term frequencies
        return doc / total_terms if total_terms > 0 else np.zeros_like(doc)  # Normalize the vector

    vec1 = normalize(doc1)  # Normalize document 1
    vec2 = normalize(doc2)  # Normalize document 2
    
    # Check if either vector is zero (empty document)
    if np.sum(vec1) == 0 or np.sum(vec2) == 0:
        return 0  # Return zero divergence if either vector is empty
    
    # Calculate K-L divergence, avoiding division by zero
    kl_div = np.sum(np.where(vec1 > 0, vec1 * np.log(vec1 / (vec2 + 1e-10)), 0))  
    return kl_div

def create_distance_matrix(doc_matrix, distance_metric):
    num_docs = doc_matrix.shape[0]  # Number of documents
    distance_matrix = np.zeros((num_docs, num_docs))  # Initialize distance matrix

    # Calculate distances between each pair of documents
    for i in range(num_docs):
        for j in range(i + 1, num_docs):  # Compute only for j > i to avoid redundant calculations
            distance_matrix[i, j] = distance_metric(doc_matrix[i], doc_matrix[j])
            distance_matrix[j, i] = distance_matrix[i, j]  # Ensure the matrix is symmetric

    return distance_matrix

def plot_all_heatmaps(jaccard_matrix, euclidean_matrix, cosine_matrix, kl_matrix):
    # Plot heatmaps for all distance matrices
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Jaccard Distance Heatmap
    sns.heatmap(jaccard_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0, 0])
    axes[0, 0].set_title("Jaccard Distance Matrix")

    # Euclidean Distance Heatmap
    sns.heatmap(euclidean_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0, 1])
    axes[0, 1].set_title("Euclidean Distance Matrix")

    # Cosine Distance Heatmap
    sns.heatmap(cosine_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 0])
    axes[1, 0].set_title("Cosine Distance Matrix")

    # KL Divergence Heatmap
    sns.heatmap(kl_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 1])
    axes[1, 1].set_title("KL Divergence Matrix")

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()  # Display the plots

# Example usage of the functions defined above
df, term_document_matrix, vocabulary = create_term_document_matrix(folder_path)

# Create distance matrices using the defined metrics
jaccard_dist_matrix = create_distance_matrix(term_document_matrix, jaccard_distance)
euclidean_dist_matrix = create_distance_matrix(term_document_matrix, euclidean_distance)
cosine_sim_matrix = create_distance_matrix(term_document_matrix, cosine_similarity)
cosine_dist_matrix = np.clip(1 - cosine_sim_matrix, 0, 1)  # Convert cosine similarity to distance

kl_div_matrix = create_distance_matrix(term_document_matrix, KL_divergence)

# Plot all distance matrices in a 2x2 grid with annotations
plot_all_heatmaps(jaccard_dist_matrix, euclidean_dist_matrix, cosine_dist_matrix, kl_div_matrix)
