from transformers import BertLMHeadModel, BertTokenizer

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('./bert_final_tokenizer')
model = BertLMHeadModel.from_pretrained('./bert_final_model')

# Text to be input to the model
input_text = 'Harvard  Universityសាកលវិទ្យាល័យ  Harvard  គឺជា សាកលវិទ្យាល័យ ឯកជន ស្រាវជ្រាវ  Ivy  League  នៅ  Cambridge  រដ្ឋ  Massachusetts  បាន បង្កើតឡើង  1636.  ប្រវត្តិសាស្រ្ត ឥទ្ធិពល របស់ ខ្លួន និង ទ្រព្យសម្បត្តិ បាន ធ្វើឱ្យ វា មួយ នៃ សាកលវិទ្យាល័យ ដ៏ មាន កិត្យានុភាព បំផុត ក្នុង ពិភពលោក ។ដើម ឡើយ បង្កើតឡើង ដោយ សភា រដ្ឋ  Massachusetts'
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
sample_outputs = model.generate(input_ids, do_sample=True, max_length=500, num_return_sequences=3)

# Open a text file to write output
with open('bert.txt', 'w', encoding='utf-8') as file:
    file.write(f"Input: {input_text}\n")
    file.write("Generated Texts:\n")
    for i, sample_output in enumerate(sample_outputs):
        output_text = tokenizer.decode(sample_output, skip_special_tokens=True)
        file.write(f"{i}: {output_text}\n")

print("Output has been written to output.txt")
