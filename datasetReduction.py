import pandas as pd
# ✅ Load NYT CSV file through read_csv
nyt_df = pd.read_csv("./archive/nyt-metadata.csv", low_memory=False)

# ✅ Check the first few rows and columns of the dataset
print(nyt_df.head())

# ✅ Randomly select 500K samples
nyt_sampled_df = nyt_df.sample(n=6000, random_state=42)  # Ensures reproducibility

# ✅ Save the reduced dataset
nyt_sampled_df.to_csv("./archive/nyt_6k.csv", index=False)

print(f"✅ Sampled {len(nyt_sampled_df)} records and saved successfully!")
