# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.g = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.V = nn.Linear(d_model, d_internal)
        self.W = nn.Linear(d_model, num_classes)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_encode = PositionalEncoding(d_model, 20, False)

        self.trans_layer = TransformerLayer(d_model, d_internal)

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        
        inp_embedding = self.emb(indices)
        # print("EMBEDDING is ", inp_embedding)
        inp_w_pos_encoding = self.pos_encode(inp_embedding)

        # print(indices)
        # print(output)

        attention, attention_map = self.trans_layer.forward(inp_w_pos_encoding)

        attention_maps = []
        attention_maps.append(attention_map)

        for i in range(self.num_layers-1):
            attention, attention_map = self.trans_layer.forward(attention)
            attention_maps.append(attention_map)


        # print("trans layer output shape ", attention.shape, attention_map.shape)
        # print((self.W(self.g(self.V((attention))))))
            
        # print("ATTENTION is ", attention)
        # print("AFTER W ", (self.W(attention)))

        log_probs = self.log_softmax((self.W(attention)))
        # print(log_probs)
        # attention_maps = []
        # attention_maps.append(attention_map)

        return (log_probs, attention_maps)


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.d_internal = d_internal
        self.wq = nn.Linear(d_model, d_internal)
        self.wk = nn.Linear(d_model, d_internal)
        self.wv = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

        self.w1 = nn.Linear(d_model, d_internal)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(d_internal, d_model)

        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)        

    def forward(self, input_vecs):
        # self attention
        # print("input is ",input_vecs)
        query = self.wq(input_vecs)
        key = self.wk(input_vecs)
        value = self.wv(input_vecs)


        # attention
        scores = torch.matmul(query, torch.t(key)) / np.sqrt(self.d_internal)
        attention_softmax = self.softmax(scores)
        # print("ATTENTION SOFTMAX ",attention_softmax)
        # print("VALUE ", value)
        attention_output = torch.matmul(attention_softmax, value)

        # Residual connection
        res = input_vecs + attention_output

        # feedForward
        feed = self.w2(self.gelu(self.w1(res)))
        z = feed + res


        return z, attention_softmax


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
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
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    # print(train[0].input_tensor)
    # print("Input Tensor Shape:", train[0].input_tensor.shape)
    # emb = nn.Embedding(27,20)
    # output = emb(train[0].input_tensor)
    # print("Output Tensor Shape:", output.shape)
    # translayer = TransformerLayer(20,27)
    # print(translayer.forward(output))
    # model = Transformer(27,20,20,27,3,1)
    # output = model.forward(train[0].input_tensor)
    # print(output[0].shape)

    # create letter counting object
    
    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:
    model = Transformer(27,20,100,100,3,2)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # train = train[:100]

    num_epochs = 10
    for t in range(0, num_epochs):
        # print("EPOCH ",t)
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            loss = loss_fcn(model.forward(train[ex_idx].input_tensor)[0], train[ex_idx].output_tensor) # TODO: Run forward and compute loss
            # print(loss)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        # print(loss_this_epoch)
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
