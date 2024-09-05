from datasets import load_dataset
from transformers import BertTokenizerFast

# Load the EURLEX dataset
dataset = load_dataset("eurlex")

# Initialize the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Access the train split
train_split = dataset['train']

# Calculate the total number of tokens and the number of documents
total_tokens = 0
total_docs = len(train_split)

for doc in train_split:
    tokens = tokenizer.tokenize(doc['text'])
    total_tokens += len(tokens)

# Calculate the average number of tokens per document
average_tokens = total_tokens / total_docs

print(f"Average number of tokens per document: {average_tokens:.2f}")