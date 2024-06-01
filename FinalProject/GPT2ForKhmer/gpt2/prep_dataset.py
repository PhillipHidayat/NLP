from datasets import load_dataset, DatasetDict

def encode(examples):
    return tokenizer.encode_batch(examples['text'])

train_path = 'C:/Users/Sun_r/Desktop/git/GPT2ForKhmer/gpt2/data/kmwiki_data'
valid_path =  'C:/Users/Sun_r/Desktop/git/GPT2ForKhmer/gpt2/data/validation_data'

# Load your dataset
dataset = load_dataset('text', data_files={'train': train_path, 'validation': valid_path})

# Apply the tokenizer
dataset = dataset.map(encode, batched=True)