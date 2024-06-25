import torch, os
from transformers import BertTokenizer, BertForPreTraining, Trainer, TrainingArguments, DataCollatorForLanguageModeling,BertForMaskedLM
from datasets import load_dataset, DatasetDict
import logging

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = BertForMaskedLM.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


# Freeze layers
for param in model.bert.encoder.layer[:5].parameters():
    param.requires_grad = False


# Load EURLEX dataset
dataset = load_dataset('eurlex')

# Tokenize the dataset
class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
tokenizer_wrapper = TokenizerWrapper(tokenizer)

tokenized_datasets = dataset.map(tokenizer_wrapper.tokenize_function, batched=True, num_proc=os.cpu_count()-2, remove_columns=['text', 'celex_id', 'title', 'eurovoc_concepts'])


# Data Collator for MLM
mlm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./mlm_eurlex_f5",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_total_limit=2,
    save_strategy="epoch",
#    save_steps=1000,
    logging_dir="./logs",
    logging_steps=100,
    metric_for_best_model="loss",
    seed=0,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=mlm_data_collator,
)


# Train and Evaluate the Model
trainer.train()
trainer.evaluate()

# logging
for obj in trainer.state.log_history:
    logging.info(str(obj))

# Save the trained model and tokenizer
#model.save_pretrained("./mlm_eurlex")
#tokenizer.save_pretrained('./mlm_eurlex')
