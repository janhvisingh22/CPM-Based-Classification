import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load WOS dataset (assuming it's preprocessed as a list of dicts)
with open("./data/processed/WOS_preprocessed.json", "r") as f:
    wos_data = json.load(f)

# Convert to DataFrame
wos_df = pd.DataFrame(wos_data)

# Load NYT dataset (assuming it's a CSV)
nyt_df = pd.read_csv("nyt_preprocessed.csv")

# 📌 Train-Test Split for WOS
wos_train, wos_test = train_test_split(
    wos_df, test_size=0.2, stratify=wos_df["labels"], random_state=42
)

# 📌 Train-Test Split for NYT (Stratify if categorical labels exist)
if "category" in nyt_df.columns:
    nyt_train, nyt_test = train_test_split(
        nyt_df, test_size=0.2, stratify=nyt_df["category"], random_state=42
    )
else:
    nyt_train, nyt_test = train_test_split(nyt_df, test_size=0.2, random_state=42)

# Save the splits
wos_train.to_json("wos_train.json", orient="records", lines=True)
wos_test.to_json("wos_test.json", orient="records", lines=True)

nyt_train.to_csv("nyt_train.csv", index=False)
nyt_test.to_csv("nyt_test.csv", index=False)

print("✅ Train-Test Split Completed!")

import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Load NYT dataset
with open("./data/processed/NYT_preprocessed.jsonl", "r") as f:
    nyt_data = json.loads(f)

# Convert to DataFrame
nyt_df = pd.DataFrame(nyt_data)

# 📌 Train-Test Split (Stratified by `labels`)
nyt_train, nyt_test = train_test_split(
    nyt_df, test_size=0.2, stratify=nyt_df["labels"], random_state=42
)

# Save the split datasets as JSON
nyt_train.to_json("nyt_train.json", orient="records", lines=True)
nyt_test.to_json("nyt_test.json", orient="records", lines=True)

print("✅ NYT Train-Test Split Completed!")
