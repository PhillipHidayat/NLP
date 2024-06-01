import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
import pickle

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('./final_tokenizer')
model = GPT2LMHeadModel.from_pretrained('./final_model').to(device)

# Load training pickle file
filename = 'accident_docs.pkl'
with open(filename, 'rb') as f:
    data_loaded = pickle.load(f)


# Creating binary labels
data_loaded['label'] = np.where(data_loaded['cat'] == 'accident', 1, 0)
texts = data_loaded['text'].tolist()
labels = data_loaded['label'].tolist()



# Tokenize the texts
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

# Convert to tensors
inputs = torch.tensor(encodings['input_ids'])
masks = torch.tensor(encodings['attention_mask'])
labels = torch.tensor(labels)

# Create a dataset from tensors
dataset = TensorDataset(inputs, masks, labels)

# Train/validation split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 4
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

from tqdm import tqdm

model.train()

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        model.zero_grad()
       
        outputs = model(b_input_ids, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Calculate average loss over the training data.
    avg_train_loss = total_loss / len(train_loader)            
    print(f"Average Training Loss: {avg_train_loss:.2f}")

    # Validation step
    model.eval()
    eval_loss = 0
    eval_accuracy = 0

    for batch in tqdm(val_loader, desc="Validating"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # calculate the accuracy and loss here using your preferred metrics


model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_tokenizer')