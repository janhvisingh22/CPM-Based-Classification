import json

with open("data/processed/WOS_preprocessed.json", "r") as file:
    wos_data = json.load(file)
print("Sample WOS Entry:", wos_data[0])

with open("data/processed/NYT_preprocessed.jsonl", "r") as file:
    nyt_data = json.loads(file)
print("Sample NYT Entry:", nyt_data[0])

# with open("data/processed/NYT_preprocessed.json", "r", encoding="utf-8") as file:
#     nyt_data = []
#     for _ in range(100):  # Load only the first 100 entries
#         try:
#             nyt_data.append(json.loads(file.readline()))
#         except:
#             break  # Stop if end of file is reached

# print("Loaded first 100 NYT entries:", nyt_data[:5])  # Print only 5 for checking

# def read_large_json(file_path, max_entries=5):
#     with open(file_path, "r", encoding="utf-8") as file:
#         for i, line in enumerate(file):
#             if i >= max_entries:  # Limit output to avoid MemoryError
#                 break
#             print(json.loads(line))

# # Read first 5 entries from NYT dataset
# read_large_json("data/processed/NYT_preprocessed.json")