import os
from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer, RobertaTokenizer
import csv
import numpy as np

# data directory and the output directory
data_directory = "../data/game_reviews"
output_directory = "../data/tokenized_reviews"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Define the tokenizers
tokenizers = {
    'bert': BertTokenizer.from_pretrained('bert-base-uncased'),
    'gpt2': GPT2Tokenizer.from_pretrained('gpt2'),
    't5': T5Tokenizer.from_pretrained('t5-small'),
    'roberta': RobertaTokenizer.from_pretrained('roberta-base')
}

# Get a list of all CSV files in the data directory
csv_files = [os.path.join(data_directory, file) for file in os.listdir(data_directory) if file.endswith('.csv')]

# Process each tokenizer
for tokenizer_name, tokenizer in tokenizers.items():
    combined_data = []

    # Process each CSV file
    for file in csv_files:
        with open(file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                review_id = row[0]
                good_review_text = row[2]
                bad_review_text = row[3]
                tokenized_good_review = tokenizer.encode(good_review_text.lower(), add_special_tokens=True)
                tokenized_bad_review = tokenizer.encode(bad_review_text.lower(), add_special_tokens=True)
                full_tokenized_review = tokenized_good_review + tokenized_bad_review
                combined_data.append([review_id, full_tokenized_review])

    # Determine the maximum length for padding
    max_len = max([len(row[1]) for row in combined_data])
    for row in combined_data:
        row[1] += [0] * (max_len - len(row[1]))  # other option tested in padding is -inf

    # Write the combined data to a new CSV file for the current tokenizer
    output_file = os.path.join(output_directory, f'{tokenizer_name}_tokenizer_reviews.csv')
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['review_id'] + list(range(max_len)))
        for row in combined_data:
            # Remove square brackets and split the flattened list
            row = [row[0]] + str(row[1])[1:-1].split(", ")
            row = np.array(row)
            writer.writerow(row.flatten())

    print(f"Tokenized data saved to {output_file}")
