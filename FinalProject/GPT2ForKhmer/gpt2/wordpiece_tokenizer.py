from tokenizers import BertWordPieceTokenizer
import os

# Ensure the model directory exists before starting the tokenizer training
model_dir = f'wordpiece_khmer_tokenizer'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer()

# Define a function to get all dataset paths
def get_dataset_paths(data_folder):
    dataset_paths = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):  # Filter for only .txt files
            file_path = os.path.join(data_folder, filename)
            dataset_paths.append(file_path)
    return dataset_paths

# Get all dataset paths from the data folder
dataset_paths = get_dataset_paths(r'./kmwiki_data')

# Customize training with UTF-8 encoding by default
tokenizer.train(
    files=dataset_paths,
    vocab_size=50257,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

# Save the tokenizer model to the prepared directory
tokenizer.save_model(model_dir)