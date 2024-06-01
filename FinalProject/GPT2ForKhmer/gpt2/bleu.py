import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import corpus_bleu
import re

# Load the pretrained GPT-2 model and tokenizer
model_path = "final_model"
tokenizer_path = "final_tokenizer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

# Load the reference human-generated text from the file
reference_file = "./oscar/oscar_kh_86.txt"
with open(reference_file, "r", encoding="utf-8") as f:
    reference_text = f.readlines()

# # Generate text using your model
# # Replace "prompt" with your desired prompt or input text
# prompt = "កម្មវិធី អាពាហ៍ពិពាហ៍ ដ៏ អស្ចារ្យ របស់ តារាចម្រៀង រូប ស្រស់  កញ្ញា  សុគន្ធ  និសា  ជាមួយនឹង"
# input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
# output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# # Tokenize the reference text and generated text
# reference_tokens = [tokenizer.tokenize(sentence.strip()) for sentence in reference_text]
# generated_tokens = tokenizer.tokenize(generated_text.strip())

# # Calculate BLEU score
# bleu_score = corpus_bleu([[tokens] for tokens in reference_tokens], [generated_tokens])
# print("BLEU Score:", bleu_score)

counter = 0
bleuSum = 0
for sentence in reference_text:
    prompt = sentence[0:int(len(sentence)/4)]
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(len(sentence))
    reference_tokens = tokenizer.tokenize(sentence.strip())
    print(len(reference_tokens))
    output = model.generate(input_ids, max_length=len(reference_tokens), num_return_sequences=1, temperature=0.7)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("OUTPUT",len(generated_text))
    print("INPUT",len(sentence))
    
    generated_tokens = tokenizer.tokenize(generated_text.strip())

    # Calculate BLEU score
    bleu_score = corpus_bleu([[tokens] for tokens in reference_tokens], [generated_tokens])
    print("BLEU Score:", bleu_score)
    counter += 1
    bleuSum += bleu_score

print("AVG BLEU: ", bleuSum/counter)