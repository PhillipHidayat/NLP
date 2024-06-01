import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define a custom dataset class for prediction
class SentimentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
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
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }

# Example texts to predict
texts_to_predict = [
  "កង្កែប មួយ ក្បាល ត្រូវ ម៉ូតូ បុក",
  "ខ្ញុំ បាន បុក ឡាន របស់ នរណា ម្នាក់ កាល ពី ឆ្នាំ មុន",
  "ដើម ឈើ រលំ សង្កត់ លើ សត្វ ស្វា នៅ ក្នុង ព្រៃ",
  "ខ្ញុំចតឡានរបស់ខ្ញុំនៅចំណត"
]

# Load fine-tuned GPT-2 tokenizer and model
tokenizer = BertTokenizer.from_pretrained('./bert_fine_tuned_tokenizer')
model = BertForSequenceClassification.from_pretrained('./bert_fine_tuned_model')

# Define max sequence length
max_length = 128

# Create dataset for prediction
predict_dataset = SentimentDataset(texts_to_predict, tokenizer, max_length)

# Define DataLoader object
predict_loader = DataLoader(predict_dataset, batch_size=1)

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prediction loop
model.eval()
for batch in predict_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    print("Predicted class:", predicted_class)
