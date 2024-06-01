# models.py

import numpy as np
import torch
import torch.nn as nn
import utils
import random
from torch import optim


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)

# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.x = num_positions
        self.d_model = d_model
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

        self.pe = np.zeros(num_positions*d_model).reshape(num_positions, d_model) 

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        # print(x)
        input_size = x.shape[-2]
        # print(input_size)
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            # print("in pos ecnode class indices to embed ", indices_to_embed, self.x)
            return x + self.emb(indices_to_embed)
        
    def encode(self, x):
        for i in np.arange(self.x):
            for d in np.arange(self.d_model//2):
                theta = i / (200 ** ((2*d/self.d_model)))
                self.pe[i, 2*d] = np.sin(theta)
                self.pe[i, 2*d+1] = np.cos(theta)

        input_size = x.shape[-2]
        # print(input_size)
        
        return self.forward(x + torch.LongTensor(self.pe[0:input_size]))

        



class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_size, d_model, chunk_size, num_positions, nhead, num_layers, dropout, vocab_index:utils.Indexer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions, False)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, activation="gelu")
        self.transformerEncoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.indexer = vocab_index
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.W = nn.Linear(d_model, vocab_size)
        self.mask = (torch.triu(torch.ones(d_model, d_model)) == 1).transpose(0, 1).float()
        self.mask = self.mask.masked_fill(self.mask == 0, float('-inf')).masked_fill(self.mask == 1, float(0.0))
        # nn.init.xavier_uniform_(self.W.weight)

        # print("NUM POS ", num_positions)
        

    def forward(self, indices):
        # print("indices ", indices)
        # embedding layer
        emb = self.embedding(indices)
        # position encoding layer
        pos_encode = self.positional_encoding.forward(emb)
        # transformer layer
        attention = self.transformerEncoder.forward((pos_encode),mask=self.mask, is_causal=True)
        # print("FORWARD ATT ", attention)
        output = self.logsoftmax(self.W(attention))

        return output

    def get_next_char_log_probs(self, context):

        input_index = []
        for char in context:
            index = self.indexer.index_of(char)
            input_index.append(index)
        input_tensor = torch.LongTensor(input_index)

        # print("in get next forward input ", input_tensor)

        attention = self.forward(input_tensor)
        # print("ATTENTION ", attention)

        # get log probs only for next chars
        log_probs = attention[-1,:]
        # print("LOG Probs ", log_probs)

        self.transformerEncoder.eval()

        return torch.flatten(log_probs).detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        # print("NEXT CHAR ", next_chars)
        # print("LEN ", len(next_chars))
        # print("Context ", context)
        # print("LEN Context ", len(context))

        if (len(context) == 0):
            context = " "

        log_probs = self.get_next_char_log_probs(context)

        log_prob_sum = 0.0

        for char in next_chars:            
            index = self.indexer.index_of(char)
            # print("index of char ", char, " is ", index)
            log_probs = self.get_next_char_log_probs(context)
            log_prob_sum += log_probs[index]
            context += char
        # print("LOG SUM ", log_prob_sum)

        self.transformerEncoder.eval()

        return log_prob_sum

        


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    chunk_size = 200

    model = NeuralLanguageModel(27,500,chunk_size,500,2, 10,dropout=0.1,vocab_index=vocab_index)
    model.transformerEncoder.zero_grad()
    model.transformerEncoder.train()
    optimizer = optim.Adam(model.transformerEncoder.parameters(), lr=1e-5)
    train = []
    gold = []

    

    for i in range(100, 499):
        train.append(train_text[i*chunk_size:(i+1)*chunk_size])
        gold.append(train_text[i*chunk_size+1:(i+1)*chunk_size+1])



    num_epochs =5
    for t in range(0, num_epochs):
        # print("EPOCH ",t)
        loss_this_epoch = 0.0
        random.seed(t)
        # random.shuffle(train_text) 
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            input_indices = get_indices(train[ex_idx], vocab_index)
            gold_indices = get_indices(gold[ex_idx], vocab_index)
            log_probs = model.forward(input_indices)
            loss = loss_fcn(log_probs[0], gold_indices[0])
            for i in range(1,chunk_size):
                loss += loss_fcn(log_probs[i], gold_indices[i]) # TODO: Run forward and compute loss
            # print(loss)
            loss = torch.divide(loss, chunk_size)
            model.transformerEncoder.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        # print(loss_this_epoch)
    model.transformerEncoder.eval()
    return model

def get_indices(list, indexer:utils.Indexer):
    indices = []
    for char in list:
        indices.append(indexer.index_of(char))
    return torch.LongTensor(indices)
        
