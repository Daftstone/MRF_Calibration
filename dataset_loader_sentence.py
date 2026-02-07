from parse import args
import random
import datasets
import tqdm
import pandas as pd
import re
from nltk.tokenize import sent_tokenize

# you can add more datasets here and write your own dataset parsing function
DATASETS = ['TruthfulQA', 'SQuAD1', 'NarrativeQA', "Essay", "Reuters", "WP"]


def process_spaces(text):
    return text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def process_text_truthfulqa_adv(text):
    if "I am sorry" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    if "as an AI language model" in text or "As an AI language model" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    return text


def check_period(texts):
    for i in range(len(texts)):
        if texts[i][-1] != ".":
            texts[i] = str(texts[i]) + "."
    return texts


def check_period_single(texts):
    if texts[-1] != ".":
        texts += "."
    return texts


def preprocess(text):
    if (args.sentence_length == -1):
        return [text]

    text_split_temp = sent_tokenize(text)

    text_split = []
    for i in range(0, len(text_split_temp), args.sentence_length):
        if (i + args.sentence_length <= len(text_split_temp)):
            text_split.append(" ".join(text_split_temp[i:i + args.sentence_length]))

    text_list = []
    for cur_text in text_split:
        if (len(cur_text.split(" ")) >= 5):
            text_list.append(cur_text)
    return text_list


def load(name, detectLLM):
    if name in ["Essay", 'Reuters']:

        f = pd.read_csv(f"datasets/{name}_LLMs.csv")
        a_human_temp = f["human"].tolist()
        a_chat_temp = f[f'{detectLLM}'].fillna("").tolist()

        a_human = []
        a_chat = []
        for text_human_temp in a_human_temp:
            a_human += preprocess(text_human_temp)
        for text_chat_temp in a_chat_temp:
            a_chat += preprocess(text_chat_temp)

        random.shuffle(a_human)
        random.shuffle(a_chat)

        maxlen = min(len(a_human), len(a_chat))
        a_human = a_human[:maxlen]
        a_chat = a_chat[:maxlen]
        res = []
        for i in range(len(a_human)):
            if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
                res.append([a_human[i], a_chat[i]])

        res = res[:10000]

        data_new = {
            'train': {
                'text': [],
                'label': [],
            },
            'test': {
                'text': [],
                'label': [],
            },
            'val': {
                'text': [],
                'label': [],
            }

        }

        index_list = list(range(len(res)))
        random.shuffle(index_list)

        total_num = len(res)
        for i in tqdm.tqdm(range(total_num), desc="parsing data"):
            if i < total_num * 0.1:
                data_partition = 'train'
            elif i >= total_num * 0.1 and i < total_num * 0.55:
                data_partition = 'val'
            else:
                data_partition = 'test'
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][0]))
            data_new[data_partition]['label'].append(0)
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][1]))
            data_new[data_partition]['label'].append(1)
        return data_new
    elif name in ["DetectRL"]:
        import json
        with open('datasets/DetectRL/Multi_LLM/multi_llms_%s_train.json' % detectLLM, 'r', encoding='utf-8') as file:
            data = json.load(file)
        a_human_temp = []
        a_chat_temp = []
        for i in range(len(data)):
            if (data[i]['label'] == 'human'):
                a_human_temp += preprocess(data[i]['text'])
            elif (data[i]['label'] == 'llm'):
                a_chat_temp += preprocess(data[i]['text'])

        a_human = []
        a_chat = []
        for text_human_temp in a_human_temp:
            a_human += preprocess(text_human_temp)
        for text_chat_temp in a_chat_temp:
            a_chat += preprocess(text_chat_temp)

        random.shuffle(a_human)
        random.shuffle(a_chat)

        maxlen = min(len(a_human), len(a_chat))
        a_human = a_human[:maxlen]
        a_chat = a_chat[:maxlen]

        res = []
        for i in range(len(a_human)):
            if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
                res.append([a_human[i], a_chat[i]])

        res = res[:10000]

        data_new = {
            'train': {
                'text': [],
                'label': [],
            },
            'test': {
                'text': [],
                'label': [],
            },
            'val': {
                'text': [],
                'label': [],
            }

        }

        index_list = list(range(len(res)))
        random.shuffle(index_list)

        total_num = len(res)
        for i in tqdm.tqdm(range(total_num), desc="parsing data"):
            if i < total_num * 0.1:
                data_partition = 'train'
            elif i >= total_num * 0.1 and i < total_num * 0.55:
                data_partition = 'val'
            else:
                data_partition = 'test'
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][0]))
            data_new[data_partition]['label'].append(0)
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][1]))
            data_new[data_partition]['label'].append(1)
        return data_new
    elif name in ["Mix"]:
        import json
        with open('datasets/DetectRL/Data_Mixing/llm_centered_mixing_train.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        a_human_temp = []
        a_chat_temp = []
        for i in range(len(data)):
            if (data[i]['label'] == 'human'):
                a_human_temp += preprocess(data[i]['text'])
            elif (data[i]['label'] == 'llm'):
                a_chat_temp += preprocess(data[i]['text'])

        a_human = []
        a_chat = []
        for text_human_temp in a_human_temp:
            a_human += preprocess(text_human_temp)
        for text_chat_temp in a_chat_temp:
            a_chat += preprocess(text_chat_temp)

        random.shuffle(a_human)
        random.shuffle(a_chat)

        maxlen = min(len(a_human), len(a_chat))
        a_human = a_human[:maxlen]
        a_chat = a_chat[:maxlen]
        res = []
        for i in range(len(a_human)):
            if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
                res.append([a_human[i], a_chat[i]])

        res = res[:10000]

        data_new = {
            'train': {
                'text': [],
                'label': [],
            },
            'test': {
                'text': [],
                'label': [],
            },
            'val': {
                'text': [],
                'label': [],
            }

        }

        index_list = list(range(len(res)))
        random.shuffle(index_list)

        total_num = len(res)
        for i in tqdm.tqdm(range(total_num), desc="parsing data"):
            if i < total_num * 0.1:
                data_partition = 'train'
            elif i >= total_num * 0.1 and i < total_num * 0.55:
                data_partition = 'val'
            else:
                data_partition = 'test'
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][0]))
            data_new[data_partition]['label'].append(0)
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][1]))
            data_new[data_partition]['label'].append(1)
        return data_new
    elif name in ["dipper", 'polish', 'back_translation']:
        import json
        with open('datasets/DetectRL/Paraphrase_Attacks/paraphrase_%s_llm_train.json' % name, 'r',
                  encoding='utf-8') as file:
            data = json.load(file)

        a_human_temp = []
        a_chat_temp = []
        for i in range(len(data)):
            if (data[i]['label'] == 'human'):
                a_human_temp += preprocess(data[i]['text'])
            elif (data[i]['label'] == 'llm'):
                a_chat_temp += preprocess(data[i]['text'])

        a_human = []
        a_chat = []
        for text_human_temp in a_human_temp:
            a_human += preprocess(text_human_temp)
        for text_chat_temp in a_chat_temp:
            a_chat += preprocess(text_chat_temp)

        random.shuffle(a_human)
        random.shuffle(a_chat)

        maxlen = min(len(a_human), len(a_chat))
        a_human = a_human[:maxlen]
        a_chat = a_chat[:maxlen]

        res = []
        for i in range(len(a_human)):
            if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
                res.append([a_human[i], a_chat[i]])

        res = res[:10000]

        data_new = {
            'train': {
                'text': [],
                'label': [],
            },
            'test': {
                'text': [],
                'label': [],
            },
            'val': {
                'text': [],
                'label': [],
            }

        }

        index_list = list(range(len(res)))
        random.shuffle(index_list)

        total_num = len(res)
        for i in tqdm.tqdm(range(total_num), desc="parsing data"):
            if i < total_num * 0.1:
                data_partition = 'train'
            elif i >= total_num * 0.1 and i < total_num * 0.55:
                data_partition = 'val'
            else:
                data_partition = 'test'
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][0]))
            data_new[data_partition]['label'].append(0)
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][1]))
            data_new[data_partition]['label'].append(1)
        return data_new
    elif name in ["Cross"]:
        import json
        with open('datasets/DetectRL/Multi_Domain/multi_domains_%s_train.json' % detectLLM, 'r',
                  encoding='utf-8') as file:
            data = json.load(file)
        a_human_temp = []
        a_chat_temp = []
        for i in range(len(data)):
            if (data[i]['label'] == 'human'):
                a_human_temp += preprocess(data[i]['text'])
            elif (data[i]['label'] == 'llm'):
                a_chat_temp += preprocess(data[i]['text'])

        a_human = []
        a_chat = []
        for text_human_temp in a_human_temp:
            a_human += preprocess(text_human_temp)
        for text_chat_temp in a_chat_temp:
            a_chat += preprocess(text_chat_temp)

        random.shuffle(a_human)
        random.shuffle(a_chat)

        maxlen = min(len(a_human), len(a_chat))
        a_human = a_human[:maxlen]
        a_chat = a_chat[:maxlen]
        res = []
        for i in range(len(a_human)):
            if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
                res.append([a_human[i], a_chat[i]])

        res = res[:10000]

        data_new = {
            'train': {
                'text': [],
                'label': [],
            },
            'test': {
                'text': [],
                'label': [],
            },
            'val': {
                'text': [],
                'label': [],
            }

        }

        index_list = list(range(len(res)))
        random.shuffle(index_list)

        total_num = len(res)
        for i in tqdm.tqdm(range(total_num), desc="parsing data"):
            if i < total_num * 0.1:
                data_partition = 'train'
            elif i >= total_num * 0.1 and i < total_num * 0.55:
                data_partition = 'val'
            else:
                data_partition = 'test'
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][0]))
            data_new[data_partition]['label'].append(0)
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][1]))
            data_new[data_partition]['label'].append(1)
        return data_new
    else:
        raise ValueError(f'Unknown dataset {name}')
