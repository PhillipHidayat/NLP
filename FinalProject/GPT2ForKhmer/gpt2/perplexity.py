import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define a custom dataset to handle the text data
class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        self.examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append(tokenized_text[i:i+block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx])

# Load the pretrained model and tokenizer
final_model = GPT2LMHeadModel.from_pretrained("final_model")
final_tokenizer = GPT2Tokenizer.from_pretrained("final_tokenizer")

# Define the parameters
block_size = 128  # You can adjust this based on your model's maximum context window

# Create an instance of the dataset
dataset = TextDataset(final_tokenizer, "./oscar/oscar_kh_86.txt", block_size)

# Function to calculate perplexity
def calculate_perplexity(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    # lossList = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[:, :-1], batch[:, 1:]
            outputs = model(inputs)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += labels.numel()
            print("LOSS", loss)

    perplexity = np.exp(total_loss / total_tokens)
    return perplexity

# Create a dataloader for the dataset
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Calculate perplexity
perplexity = calculate_perplexity(final_model, dataloader)

print("Perplexity:", perplexity)
