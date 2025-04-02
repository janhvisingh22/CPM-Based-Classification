# This script preprocesses the Web of Science (WOS) and New York Times (NYT) datasets.
import os
import json
import pandas as pd
import numpy as np
import ast  # Safer alternative to eval()
from transformers import DistilBertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# 🔹 Initialize DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# # Paths to WOS dataset files
WOS_TEXT_FILE = "./WebOfScience/WOS46985/X.txt"    # Textual content
WOS_LABEL_FILE = "./WebOfScience/WOS46985/Y.txt"   # Primary labels
WOS_LEVEL1_FILE = "./WebOfScience/WOS46985/YL1.txt" # First-level hierarchical labels
WOS_LEVEL2_FILE = "./WebOfScience/WOS46985/YL2.txt" # Second-level hierarchical labels

# Path to NYT dataset file
NYT_CSV_FILE = "./archive/nyt_500k.csv"  # NYT dataset in CSV format

# Load text data from a file
def load_texts(path):
    with open(path, "r", encoding="utf-8") as file:
        texts = file.readlines()
    return [text.strip() for text in texts]

# Load labels from a file
def load_labels(path):
    with open(path, "r", encoding="utf-8") as file:
        labels = file.readlines()
    return [int(label.strip()) for label in labels]

# Load NYT dataset from CSV
def load_nyt_dataset(path):
    df = pd.read_csv(path, dtype={"print_page": str}, low_memory=False)

    texts = df["abstract"].fillna("").tolist()  # Use abstract as main text
    labels = df["section_name"].fillna("Unknown").tolist()
    level1_labels = df["section_name"].fillna("Unknown").tolist()
    level2_labels = df["subsection_name"].fillna("Unknown").tolist()
    
    # Extract keywords safely
    keywords = []
    for kw_list in df["keywords"]:
        try:
            parsed_keywords = [kw['value'] for kw in ast.literal_eval(kw_list)] if isinstance(kw_list, str) else []
        except:
            parsed_keywords = []
        keywords.append(parsed_keywords[:4])  # Keep top 4 keywords
    
    return texts, labels, level1_labels, level2_labels, keywords

# 🔹 Tokenize text using DistilBERT
def tokenize_text(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    
    # Convert tensors to lists for JSON compatibility
    return {key: value.tolist() if hasattr(value, "tolist") else value for key, value in tokens.items()}

# Extract top keywords using TF-IDF
def extract_keywords(texts, num_keywords=4):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(texts)
    
    keywords_per_doc = []
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    for i in range(X.shape[0]):
        sorted_indices = np.argsort(X[i].toarray()).flatten()[::-1]
        keywords = feature_names[sorted_indices][:num_keywords].tolist()
        keywords_per_doc.append(keywords)
    
    return keywords_per_doc

# 🔹 Process WOS dataset
def process_wos_dataset():
    texts = load_texts(WOS_TEXT_FILE)
    labels = load_labels(WOS_LABEL_FILE)
    level1_labels = load_labels(WOS_LEVEL1_FILE)
    level2_labels = load_labels(WOS_LEVEL2_FILE)
    
    assert len(texts) == len(labels) == len(level1_labels) == len(level2_labels), "WOS dataset size mismatch!"
    
    tokenized_texts = [tokenize_text(text) for text in texts]
    keywords = extract_keywords(texts)
    
    processed_data = [{"text": text, "tokens": tokens, "labels": lbl, "level1": lvl1, "level2": lvl2, "keywords": kw}
                      for text, tokens, lbl, lvl1, lvl2, kw in zip(texts, tokenized_texts, labels, level1_labels, level2_labels, keywords)]
    
    os.makedirs("data/processed", exist_ok=True)
    save_path = "data/processed/WOS_preprocessed_new.json"
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(processed_data, file, indent=4)
    print(f"✅ Processed WOS data saved to {save_path}")

# 🔹 Process NYT dataset
def process_nyt_dataset():
    texts, labels, level1_labels, level2_labels, keywords = load_nyt_dataset(NYT_CSV_FILE)

    if len(texts) == 0:
        print("🚨 NYT dataset is empty! Check the CSV file path and structure.")
        return

    save_path = "data/processed/NYT_preprocessed_new.jsonl"
    with open(save_path, "w", encoding="utf-8") as file:
        for text, lbl, lvl1, lvl2, kw in zip(texts, labels, level1_labels, level2_labels, keywords):
            tokens = tokenize_text(text)
            json.dump({"text": text, "tokens": tokens, "labels": lbl, "level1": lvl1, "level2": lvl2, "keywords": kw}, file)
            file.write("\n")  # Newline for JSONL format

    print(f"✅ Processed NYT data saved to {save_path}")

# Run preprocessing
if __name__ == "__main__":
    process_wos_dataset()
    process_nyt_dataset()

