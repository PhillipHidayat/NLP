import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
import pickle

# Define a custom dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load training pickle file
filename = 'accident_docs.pkl'
with open(filename, 'rb') as f:
    data_loaded = pickle.load(f)

# Creating binary labels
data_loaded['label'] = np.where(data_loaded['cat'] == 'accident', 1, 0)
texts = data_loaded['text'].tolist()
labels = data_loaded['label'].tolist()

# Load pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('./final_tokenizer')

# Define max sequence length
max_length = 128

# Create datasets
dataset = SentimentDataset(texts, labels, tokenizer, max_length)

# Split dataset into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load pre-trained GPT-2 model trained on Khmer
model = GPT2ForSequenceClassification.from_pretrained('./final_model')

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):  # 3 epochs
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Average Loss: {avg_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} Validation Average Loss: {avg_val_loss:.4f}")

# Save fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_tokenizer')
