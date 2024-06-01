## Requirement:
- python 3.6+

## Set up

# Windows
python -m venv gpt2-env
source gpt2-env\Scripts\activate

# Mac or Linux
python3 -m venv gpt2-env
source gpt2-env/bin/activate


# Install required libraries
```bash
pip install transformers torch
pip install accelerate -U
```


# Training
To start pre-training the GPT-2 model, we have to start by training our Tokenizer. To do that,
have *training* and *validation* dataset (as .txt files) inside gpt2/kmwiki_data and gpt2/validation_data
respectively.

Then, run ```python tokenizer.py``` to train the tokenizer.

Finally, you can train the GPT-2 by running ```python pretrain-gpt2.py```

