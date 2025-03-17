import os
import pandas as pd
import json
from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Paths to the datasets
