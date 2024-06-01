from datasets import load_dataset, Dataset
import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load the tokenizer
tokenizer_path = 'khmer_tokenizer'
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
})

# Function to tokenize the dataset
def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')


data_files = {
    'train': ['C:/Users/Sun_r/Desktop/git/GPT2ForKhmer/gpt2/data/*.txt'],
    'validation': ['C:/Users/Sun_r/Desktop/git/GPT2ForKhmer/gpt2/validation_data/*.txt']
}

dataset = load_dataset('text', data_files=data_files)

# Apply the tokenizer to the dataset
dataset = dataset.map(encode, batched=True)

# Set dataset format for PyTorch
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Model and training configuration
model_config = GPT2Config.from_pretrained('gpt2', vocab_size=len(tokenizer))
model = GPT2LMHeadModel(model_config)

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# Initialize and start the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation']
)

trainer.train()

# Save the model and tokenizer
model.save_pretrained('./final_model')
tokenizer.save_pretrained('./final_tokenizer')
