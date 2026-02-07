import random
import re

import numpy as np
import torch
import tqdm
import argparse
import json
import transformers
import pickle

from transformers import AutoTokenizer, AutoModelForCausalLM


class PrefixSampler:
    def __init__(self, args):
        self.args = args
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "save_models/" + 'gpt2').to(args.DEVICE)
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            "save_models/" + 'gpt2')
        if self.base_tokenizer.pad_token_id is None:
            self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id

    def _sample_from_model(self, texts, min_words=55, truncate_ratio=0.5):
        texts = [t.split(' ') for t in texts]
        texts = [' '.join(t[: int(len(t) * truncate_ratio)]) for t in texts]
        all_encoded = self.base_tokenizer(texts, return_tensors="pt", padding=True).to(
            self.args.DEVICE)

        self.base_model.eval()
        decoded = ['' for _ in range(len(texts))]

        tries = 0
        m = 0
        while m < min_words:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs = {'temperature': 1.}
            if self.args.do_top_p:
                sampling_kwargs['top_p'] = 0.96
            elif self.args.do_top_k:
                sampling_kwargs['top_k'] = 40
            min_length = 150
            try:
                outputs = self.base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True,
                                                   **sampling_kwargs, pad_token_id=self.base_tokenizer.eos_token_id,
                                                   eos_token_id=self.base_tokenizer.eos_token_id)
            except:
                try:
                    outputs = self.base_model.generate(**all_encoded, min_length=min_length, max_length=500,
                                                       do_sample=True,
                                                       **sampling_kwargs, pad_token_id=self.base_tokenizer.eos_token_id,
                                                       eos_token_id=self.base_tokenizer.eos_token_id)
                except:
                    outputs = self.base_model.generate(**all_encoded, min_length=min_length, max_length=1024,
                                                       do_sample=True,
                                                       **sampling_kwargs, pad_token_id=self.base_tokenizer.eos_token_id,
                                                       eos_token_id=self.base_tokenizer.eos_token_id)
            decoded = self.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            m = min(len(x.split()) for x in decoded)
            tries += 1

        return decoded

    def generate_samples(self, raw_data, batch_size):
        # trim to shorter length
        def _trim_to_shorter_length(texta, textb):
            # truncate to shorter of o and s
            shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
            texta = ' '.join(texta.split(' ')[:shorter_length])
            textb = ' '.join(textb.split(' ')[:shorter_length])
            return texta, textb

        def _truncate_to_substring(text, substring, idx_occurrence):
            # truncate everything after the idx_occurrence occurrence of substring
            assert idx_occurrence > 0, 'idx_occurrence must be > 0'
            idx = -1
            for _ in range(idx_occurrence):
                idx = text.find(substring, idx + 1)
                if idx == -1:
                    return text
            return text[:idx]

        data = {
            "original": [],
            "sampled": [],
        }

        for batch in range((len(raw_data) - 1) // batch_size + 1):
            # print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
            original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
            sampled_text = self._sample_from_model(original_text,
                                                   min_words=10,
                                                   truncate_ratio=0.5)

            for o, s in zip(original_text, sampled_text):
                o, s = _trim_to_shorter_length(o, s)

                # add to the data
                data["original"].append(o)
                data["sampled"].append(s)

        return data


def get_likelihood(logits, labels, pad_index):
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels).cpu().numpy()
    return log_likelihood


def get_log_prob(sampler, text):
    tokenized = sampler.base_tokenizer(text, return_tensors="pt", max_length=512).to(
        sampler.args.DEVICE)
    labels = tokenized.input_ids[:, 1:]
    with torch.no_grad():
        logits_score = sampler.base_model(**tokenized).logits[:, :-1]
        return get_likelihood(logits_score, labels, sampler.base_tokenizer.pad_token_id)


def get_log_probs(sampler, texts):
    tokenized = sampler.base_tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True,
                                       max_length=512).to(sampler.args.DEVICE)
    labels = tokenized.input_ids[:, 1:]
    with torch.no_grad():
        logits_score = sampler.base_model(**tokenized).logits[:, :-1]
        lprobs = get_likelihood(logits_score, labels, sampler.base_tokenizer.pad_token_id)
    return lprobs


def get_regen_samples(sampler, text):
    data = [text] * 10
    data = sampler.generate_samples(data, batch_size=sampler.args.batch_size)
    return data['sampled']


def get_dna_gpt(sampler, text):
    lprob = get_log_prob(sampler, text)
    regens = get_regen_samples(sampler, text)
    lprob_regens = get_log_probs(sampler, regens)
    return lprob, lprob_regens


def run_dna_experiment(
        args,
        data,
        method="DNAGPT"):
    sampler = PrefixSampler(args)

    torch.manual_seed(0)
    np.random.seed(0)

    for nn in ['train', 'val', 'test']:
        train_text = data[nn]['text']
        train_label = data[nn]['label']

        lprob_list = []
        lprob_regens_list = []
        label_list = []
        for i in tqdm.tqdm(range(len(train_text))):
            criterion_fn = get_dna_gpt
            try:
                text = train_text[i]
                label = train_label[i]
                text = " ".join(text.split()[:500])
                lprob, lprob_regens = criterion_fn(sampler, text)
                lprob_list.append(lprob)
                lprob_regens_list.append(lprob_regens)
                label_list.append(label)
            except:
                continue

        with open("datasets/embedding/x_%s_lprob_%s_%s_%s_%d_%d.pkl" % (
                nn, args.dataset, args.detectLLM, args.method, args.sentence_length, args.iter), 'wb') as f:
            pickle.dump(lprob_list, f)
        with open("datasets/embedding/x_%s_lprob_regens_%s_%s_%s_%d_%d.pkl" % (
                nn, args.dataset, args.detectLLM, args.method, args.sentence_length, args.iter), 'wb') as f:
            pickle.dump(lprob_regens_list, f)
        with open("datasets/embedding/y_%s_%s_%s_%s_%d_%d.pkl" % (
                nn, args.dataset, args.detectLLM, args.method, args.sentence_length, args.iter), 'wb') as f:
            pickle.dump(label_list, f)
    exit(0)
