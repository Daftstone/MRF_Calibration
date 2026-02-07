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
from sentence_transformers import SentenceTransformer

import re
from nltk.tokenize import sent_tokenize


class Fast(nn.Module):
    def __init__(self, DEVICE, save_path, load_path, pos_bit=1, method='EMV'):
        super(Fast, self).__init__()
        self.embedding_model = SentenceTransformer("./save_models/LUAR-MUD-sentence-transformers")

        # 定义第一层
        self.fc1 = nn.Linear(512, 128)
        # 定义第二层
        self.fc2 = nn.Linear(512, 64)
        # 定义输出层
        self.fc3 = nn.Linear(512, 2)

        self.loss_fct = CrossEntropyLoss()

        params_to_optimize = list(self.embedding_model.parameters()) + list(self.fc1.parameters()) + list(
            self.fc2.parameters()) + list(
            self.fc3.parameters())
        optimizer_grouped_parameters = [
            {'params': [p for p in params_to_optimize], 'weight_decay': 0.01}
        ]
        if(args.dataset == "Essay" or args.dataset == "SQuAD1"):
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
        if (args.dataset == 'Reuters'):
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3)
        self.pos_bit = pos_bit
        self.device = DEVICE
        self.save_path = save_path
        self.load_path = load_path

    def forward(self, x):

        outputs = self.one_forward(x)

        return outputs.softmax(-1)

    def E_step(self, x, y):

        self.optimizer.zero_grad()

        logits = self.one_forward(x)
        y = y.to(logits.device)
        self.loss = self.loss_fct(logits.view(-1, 2), y.view(-1))

    def M_step(self):
        self.loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.state_dict(),
                   f'%s/model_{args.conf_threshold}_{args.filter_threshold}_{args.sentence_num}_{args.iter}.pth' % self.save_path)
        # self.detector.save_pretrained(self.save_path)

    def load(self):
        self.load_state_dict(torch.load(
            f'%s/model_{args.conf_threshold}_{args.filter_threshold}_{args.sentence_num}_{args.iter}.pth' % self.load_path,
            map_location=self.device))
        # self.detector = transformers.AutoModelForSequenceClassification.from_pretrained(
        #     self.load_path,).to(self.device)

    def split(self, x):

        cur_x_split = sent_tokenize(x)
        sentence_list = cur_x_split

        sentence_list = []
        for x in cur_x_split:
            if (len(x) < 3 and len(sentence_list) > 0):
                sentence_list[-1] += ' ' + x
            else:
                sentence_list.append(x)

        if (args.sentence_num == 1):
            lens = 1
            sentence_list = sentence_list[:200]
        else:
            # lens = (len(sentence_list) + args.sentence_num - 1) // args.sentence_num
            lens = args.sentence_num
        nums = (len(sentence_list) - 1) // lens + 1
        paragraph_list = []
        for i in range(nums):
            begin = i * lens
            end = min(len(sentence_list), i * lens + lens)
            paragraph_list.append(" ".join(sentence_list[begin:end]))
        return paragraph_list

    def finetune(self, x, y):
        self.optimizer.zero_grad()
        encoded_input = self.tokenizer(x, return_tensors='pt', truncation=True, padding=True).to(
            self.device)
        outputs = self.detector(
            encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'], labels=y)
        loss = outputs[0]
        loss.backward()
        self.optimizer.step()

    def pretrain_forward(self, x):
        encoded_input = self.tokenizer(x, return_tensors='pt', truncation=True, padding=True).to(
            self.device)
        outputs = self.detector(
            encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])
        return outputs.logits.softmax(-1)

    def one_forward(self, x):
        with torch.no_grad():
            x = self.get_embedding(x)

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_embedding(self, x):
        """Extract the LUAR embeddings for the data.
            """
        embeddings = torch.tensor(self.embedding_model.encode(x)).to(self.device)
        return embeddings
