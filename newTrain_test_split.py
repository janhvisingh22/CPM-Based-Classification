import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load WOS dataset (assuming it's preprocessed as a list of dicts)
with open("./data/processed/WOS_preprocessed_new.json", "r") as f:
    wos_data = json.load(f)

# Convert to DataFrame
wos_df = pd.DataFrame(wos_data)

# Load NYT dataset (JSONL format)
nyt_data = []
with open("./data/processed/NYT_preprocessed_new.jsonl", "r") as f:
    for line in f:
        nyt_data.append(json.loads(line.strip()))  # Read line by line

# Convert to DataFrame
nyt_df = pd.DataFrame(nyt_data)

# 📌 Train-Test Split for WOS
wos_train, wos_test = train_test_split(
    wos_df, test_size=0.2, stratify=wos_df["labels"], random_state=42
)

# Count label occurrences
label_counts = nyt_df["labels"].value_counts()

# Filter out labels with fewer than 2 occurrences
valid_labels = label_counts[label_counts > 1].index
nyt_df_filtered = nyt_df[nyt_df["labels"].isin(valid_labels)]

# Perform stratified train-test split
nyt_train, nyt_test = train_test_split(
    nyt_df_filtered, test_size=0.2, stratify=nyt_df_filtered["labels"], random_state=42
)

print(f"✅ Train Size: {len(nyt_train)}, Test Size: {len(nyt_test)}")

# Save the split datasets as JSONL (line-separated JSON)
with open("./data/processed/nyt_train_new.jsonl", "w") as f:
    for record in nyt_train.to_dict(orient="records"):
        f.write(json.dumps(record) + "\n")

with open("./data/processed/nyt_test_new.jsonl", "w") as f:
    for record in nyt_test.to_dict(orient="records"):
        f.write(json.dumps(record) + "\n")

print("✅ NYT Train-Test Split Completed!")

# Save the splits
wos_train.to_json("./data/processed/wos_train_new.json", orient="records", lines=True)
wos_test.to_json("./data/processed/wos_test_new.json", orient="records", lines=True)