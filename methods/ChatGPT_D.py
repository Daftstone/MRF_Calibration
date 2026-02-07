from parse import args
import numpy as np
import torch
from torch import nn
import transformers
import torch
import torch.nn.functional as F
import transformers.modeling_outputs as modeling_outputs
from torch.nn import CrossEntropyLoss
from transformers import AdamW
import random

import re
from nltk.tokenize import sent_tokenize


class ChatGPT_D(nn.Module):
    def __init__(self, DEVICE, save_path, load_path, pos_bit=1):
        super(ChatGPT_D, self).__init__()
        model = 'chatgpt-detector-roberta'
        self.pos_bit = pos_bit
        self.device = DEVICE
        self.save_path = save_path
        self.load_path = load_path
        self.sent_detector = transformers.AutoModelForSequenceClassification.from_pretrained(
            "save_models/" + model,
            num_labels=2,
            ignore_mismatched_sizes=True).to(DEVICE)

        self.temperature = 1.
        self.adversarial = args.sentence_num
        self.max_token = 32 * args.sentence_length
        if (args.sentence_length == -1):
            self.max_token = 256
        self.reg = args.reg
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "save_models/" + model)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.sent_detector.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.sent_detector.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.},
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6)

    def forward(self, x):
        with torch.no_grad():
            encoded_input = self.tokenizer(x, return_tensors='pt', truncation=True, padding='max_length',
                                           max_length=self.max_token).to(self.device)
            outputs = self.sent_detector(**encoded_input)
        return outputs.logits.softmax(-1)

    def Optimize(self, x, y):
        self.optimizer.zero_grad()

        encoded_input = self.tokenizer(x, return_tensors='pt', truncation=True, padding='max_length',
                                       max_length=self.max_token).to(self.device)
        logits = self.sent_detector(**encoded_input).logits
        loss_fct = CrossEntropyLoss()
        loss1 = loss_fct(logits.view(-1, 2), y.view(-1))

        self.loss = loss1

        self.loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.state_dict(),
                   f'%s/model_{args.sentence_num}_{args.sentence_length}_{args.adversarial}_{self.reg}_{args.iter}.pth' % self.save_path)

    def load(self):
        self.load_state_dict(torch.load(
            f'%s/model_{args.sentence_num}_{args.sentence_length}_{args.adversarial}_{self.reg}_{args.iter}.pth' % self.load_path,
            map_location=self.device))

    def update_temperature(self):
        self.temperature = max(0.2, self.temperature * 0.7)
