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


class RADAR(nn.Module):
    def __init__(self, DEVICE, save_path, load_path, pos_bit=1):
        super(RADAR, self).__init__()
        model = 'RADAR'
        self.pos_bit = pos_bit
        self.device = DEVICE
        self.save_path = save_path
        self.load_path = load_path
        self.sent_detector = transformers.AutoModelForSequenceClassification.from_pretrained(
            "save_models/" + model,
            num_labels=2,
            ignore_mismatched_sizes=True).to(DEVICE)

        self.temperature = 0.5
        self.adversarial = args.sentence_num
        self.max_token = 32 * args.sentence_length
        if (args.sentence_length == -1):
            self.max_token = 128
        self.reg = args.reg
        self.para_detector = transformers.AutoModelForSequenceClassification.from_pretrained(
            "save_models/" + model,
            num_labels=2,
            ignore_mismatched_sizes=True).to(DEVICE)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "save_models/" + model)

        self.fc = nn.Sequential(
            nn.Linear(1024 * self.max_token * args.sentence_num, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        ).to(DEVICE)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.sent_detector.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.sent_detector.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.},
            # {'params': [p for n, p in self.para_detector.named_parameters() if not any(
            #     nd in n for nd in no_decay)], 'weight_decay': 0.01},
            # {'params': [p for n, p in self.para_detector.named_parameters() if any(
            #     nd in n for nd in no_decay)], 'weight_decay': 0.},
            {'params': [p for n, p in self.fc.named_parameters()], 'weight_decay': 0.01},
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6)

    def forward(self, x):
        with torch.no_grad():
            encoded_input = self.tokenizer(x, return_tensors='pt', truncation=True, padding='max_length',
                                           max_length=self.max_token).to(self.device)
            outputs = self.sent_detector(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])
        return outputs.logits.softmax(-1)

    def para_forward(self, x, y):
        with torch.no_grad():
            index_list = []
            paragraph_labels = []
            num_sentences = args.sentence_num

            encoded_input = self.tokenizer(x, return_tensors='pt', truncation=True, padding='max_length',
                                           max_length=self.max_token).to(self.device)
            logits = self.sent_detector(**encoded_input).logits

            true_labels = y.cpu().data.numpy()
            data_label_0 = np.where(true_labels == 0)[0]
            data_label_1 = np.where(true_labels == 1)[0]

            try:
                for _ in range(args.paragraph_num // 2):
                    selected_indices = random.choices(data_label_0, k=num_sentences)
                    index_list += selected_indices
                    paragraph_label = 1 if any(y[i] == 1 for i in selected_indices) else 0
                    paragraph_labels.append(paragraph_label)
                for _ in range(args.paragraph_num // 2):
                    selected_indices = random.choices(data_label_0, k=num_sentences - self.adversarial)
                    selected_indices += random.choices(data_label_1, k=self.adversarial)
                    random.shuffle(selected_indices)
                    index_list += selected_indices
                    paragraph_label = 1 if any(y[i] == 1 for i in selected_indices) else 0
                    paragraph_labels.append(paragraph_label)
            except Exception as e:
                return

            index_list = np.array(index_list)

            sub_pred = F.gumbel_softmax(logits, tau=0.2)[:, self.pos_bit]
            sub_pred = sub_pred[index_list]

            word_embedding = self.para_detector.roberta.embeddings.word_embeddings(encoded_input['input_ids'])
            word_embedding = word_embedding[index_list]
            mask = sub_pred.unsqueeze(1).unsqueeze(2)

            combined_embeddings = []
            for i in range(0, index_list.shape[0], num_sentences):
                mins = i
                maxs = i + num_sentences
                # mean_embeddings = torch.sum(word_embedding[mins:maxs] * mask[mins:maxs], dim=0) / torch.sum(
                #     mask[mins:maxs] + 1e-10)
                mean_embeddings = (word_embedding[mins:maxs] * mask[mins:maxs]).view(
                    1024 * self.max_token * args.sentence_num)
                combined_embeddings.append(mean_embeddings)
            combined_embeddings = torch.stack(combined_embeddings)  # (batch_size/3, seq_len*4, hidden_size)

            # para_pred = self.sent_detector(inputs_embeds=combined_embeddings).logits.softmax(-1).cpu().data.numpy()
            para_pred = self.fc(combined_embeddings.view(combined_embeddings.size(0), -1)).softmax(
                -1).cpu().data.numpy()

            return para_pred[:, self.pos_bit].tolist(), paragraph_labels

    def Optimize(self, x, y):
        self.optimizer.zero_grad()

        index_list = []
        paragraph_labels = []
        num_sentences = args.sentence_num

        encoded_input = self.tokenizer(x, return_tensors='pt', truncation=True, padding='max_length',
                                       max_length=self.max_token).to(self.device)
        logits = self.sent_detector(**encoded_input).logits
        loss_fct = CrossEntropyLoss()
        loss1 = loss_fct(logits.view(-1, 2), y.view(-1))

        if (args.reg == 0):
            self.loss = loss1
        else:
            preds = logits.softmax(-1).cpu().data.numpy()[:, self.pos_bit]

            true_labels = y.cpu().data.numpy()
            data_label_00 = np.where(true_labels == 0)[0]
            data_label_11 = np.where(true_labels == 1)[0]

            pred_0 = preds[data_label_00]
            pred_1 = preds[data_label_11]

            data_label_0 = list(data_label_00[np.where(pred_0 <= 1.)[0]])
            data_label_1 = list(data_label_11[np.where(pred_1 >= 0.)[0]])

            try:
                for _ in range(args.paragraph_num // 2):
                    selected_indices = random.choices(data_label_0, k=num_sentences)
                    index_list += selected_indices
                    paragraph_label = 1 if any(y[i] == 1 for i in selected_indices) else 0
                    paragraph_labels.append(paragraph_label)
                for _ in range(args.paragraph_num // 2):
                    selected_indices = random.choices(data_label_0, k=num_sentences - self.adversarial)
                    selected_indices += random.choices(data_label_1, k=self.adversarial)
                    random.shuffle(selected_indices)
                    index_list += selected_indices
                    paragraph_label = 1 if any(y[i] == 1 for i in selected_indices) else 0
                    paragraph_labels.append(paragraph_label)

                paragraph_labels = torch.tensor(paragraph_labels).to(self.device)
                index_list = np.array(index_list)

                sub_pred = F.gumbel_softmax(logits, tau=0.5)[:, self.pos_bit]
                # sub_pred = logits.softmax(-1)[:, self.pos_bit]
                sub_pred = sub_pred[index_list]

                word_embedding = self.para_detector.roberta.embeddings.word_embeddings(encoded_input['input_ids'])
                word_embedding = word_embedding[index_list]
                mask = sub_pred.unsqueeze(1).unsqueeze(2)

                combined_embeddings = []
                for i in range(0, index_list.shape[0], num_sentences):
                    mins = i
                    maxs = i + num_sentences
                    mean_embeddings = (word_embedding[mins:maxs] * mask[mins:maxs]).view(
                        1024 * self.max_token * args.sentence_num)
                    combined_embeddings.append(mean_embeddings)
                combined_embeddings = torch.stack(combined_embeddings)  # (batch_size/3, seq_len*4, hidden_size)

                para_logits = self.fc(combined_embeddings.view(combined_embeddings.size(0), -1))

                loss2 = loss_fct(para_logits.view(-1, 2), paragraph_labels.view(-1))

                self.loss = loss1 + loss2 * args.reg
            except Exception as e:
                print(e)
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
        self.temperature = max(0.2, self.temperature * 0.9)
