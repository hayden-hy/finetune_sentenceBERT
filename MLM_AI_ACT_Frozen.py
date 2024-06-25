import torch
from transformers import BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling,BertForMaskedLM   , BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup



# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained BERT model and tokenizer
model = BertForMaskedLM.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = BertTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# Freeze Layers
for param in model.bert.encoder.layer[:5].parameters():
    param.requires_grad = False

# Open the file in read mode
datapath = './ai_act_corpus.txt'
with open(datapath, 'r', encoding='utf-8') as file:
    # Read lines into a list
    dataset = file.readlines()

dataset = "".join(dataset)
# Split the dataset into train and validation sets

tokenized_dataset = tokenizer([dataset], padding='max_length', truncation=True, max_length=128)

# Data collator for MLM
mlm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# # Create DataLoaders
# train_dataloader = DataLoader(tokenized_datasets, batch_size=2, collate_fn=mlm_data_collator, shuffle=True)

from torch.utils.data import DataLoader, Dataset

class MLMDataset(Dataset):
    def __init__(self, tokenized_sentences):
        self.tokenized_sentences = tokenized_sentences
    
    def __len__(self):
        return len(list(self.tokenized_sentences.values())[0])
    
    def __getitem__(self, idx):
        # item = self.tokenized_sentences[idx]
        # Remove the batch dimension to make it compatible with DataCollatorForLanguageModeling
        # print({key: val[idx] for key, val in self.tokenized_sentences.items()})
        return {key: val[idx] for key, val in self.tokenized_sentences.items()}

train_size = 525
val_size = 50

custom_dataset = MLMDataset(tokenized_dataset)
torch.manual_seed(0)

#print(len(custom_dataset))
# train_dataset,val_dataset = torch.utils.data.random_split(custom_dataset,[train_size,val_size])
# train_dataloader = DataLoader(custom_dataset, batch_size=2, collate_fn=mlm_data_collator, shuffle=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./mlm_ai_act_f5",
    overwrite_output_dir=True,
#    eval_strategy="steps",
#    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
#    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=1,
    max_steps=80000,
    save_strategy="steps",
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    metric_for_best_model="loss",
    seed=0,
)
# Data collator for MLM
mlm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=custom_dataset,#train_dataset,
#    eval_dataset=val_dataset,
    data_collator=mlm_data_collator,
)

# Start training
trainer.train()
trainer.evaluate()
