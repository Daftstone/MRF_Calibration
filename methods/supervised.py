import numpy as np
import transformers
import torch
from tqdm import tqdm
from methods.utils import timeit, cal_metrics
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.utils.data as Data

from methods.ChatGPT_D import ChatGPT_D
from methods.MPU import MPU
from methods.RADAR import RADAR

from methods.utils import my_dataset
from parse import args
from nltk.tokenize import sent_tokenize


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


@timeit
def run_supervised_experiment(
        data,
        model,
        cache_dir,
        batch_size,
        DEVICE,
        pos_bit=0,
        finetune=False,
        num_labels=2,
        epochs=5,
        save_path=None,
        load_path="",
        **kwargs):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(
        "save_models/" + model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "save_models/" + model, cache_dir=cache_dir)

    if (len(load_path) > 0):
        detector.load_state_dict(torch.load('%s/model.pth' % load_path, map_location=DEVICE))
    if finetune:
        fine_tune_model(detector, tokenizer, data, batch_size,
                        DEVICE, pos_bit, num_labels, epochs, save_path)

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    if num_labels == 2:
        # train_preds = get_supervised_model_prediction(
        #     detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction(
            detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)
    else:
        train_preds = get_supervised_model_prediction_multi_classes(
            detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction_multi_classes(
            detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)

    predictions = {
        # 'train': train_preds,
        'test': test_preds,
    }
    # y_train_pred_prob = train_preds
    # y_train_pred = [round(_) for _ in y_train_pred_prob]
    # y_train = train_label

    y_test_pred_prob = test_preds
    y_test_pred = [round(_) for _ in y_test_pred_prob]
    y_test = test_label

    # train_res = cal_metrics(y_train, y_train_pred, y_train_pred_prob)
    test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
    acc_train, precision_train, recall_train, f1_train, auc_train = 0, 0, 0, 0, 0
    acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test = test_res
    # print(
    #     f"{model} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
    print(
        f"{model} acc_test: {acc_test}, auc_test: {auc_test}, tpr1_test: {tpr1_test}, tpr2_test: {tpr2_test}, tpr3_test: {tpr3_test}")

    # free GPU memory
    del detector
    with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'general': {
            'acc_train': acc_train,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'auc_train': auc_train,
            'acc_test': acc_test,
            'tpr1_test': tpr1_test,
            'tpr2_test': tpr2_test,
            'tpr3_test': tpr3_test,
            'auc_test': auc_test,
        }
    }


@timeit
def run_supervised_experiment_New(
        data,
        model,
        cache_dir,
        batch_size,
        DEVICE,
        pos_bit=0,
        finetune=False,
        num_labels=2,
        epochs=5,
        save_path=None,
        load_path="",
        **kwargs):
    print(f'Beginning supervised evaluation with {model}...')
    if (model == 'ChatGPT_D'):
        detector = ChatGPT_D(DEVICE, save_path, load_path, pos_bit)
    elif (model == 'MPU'):
        detector = MPU(DEVICE, save_path, load_path, pos_bit)
    elif (model == 'RADAR'):
        detector = RADAR(DEVICE, save_path, load_path, pos_bit)
    else:
        exit(0)

    if (len(load_path) > 0):
        print('load model')
        detector.load()
    if finetune:
        train_label = data['train']['label']
        if pos_bit == 0 and num_labels == 2:
            train_label = [1 if _ == 0 else 0 for _ in train_label]

        best_acc = -1
        temp_reg = args.reg
        pre_map = {'ChatGPT_D': 1, 'MPU': 1, 'RADAR': 1}
        for i in range(epochs):
            detector.train()
            if (i < pre_map[args.method]):
                args.reg = 0.
            else:
                args.reg = temp_reg

            train_dataset = my_dataset(data['train']['text'], train_label)
            loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

            pbar = tqdm(loader, desc=f"Fine-tuning: {i} epoch")
            for (batch_x, batch_y) in pbar:
                batch_y = batch_y.to(DEVICE)
                detector.Optimize(batch_x, batch_y)
            detector.eval()
            acc_train, auc_train, tpr1_train, tpr2_train, tpr3_train, acc_test, acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test = eval_data(
                data, detector, batch_size, DEVICE,
                pos_bit, pre_train=False)

            if (auc_train >= best_acc):
                print('save best')
                print('****************************\n')
                detector.save()
                best_acc = auc_train

    detector.eval()  # 0102

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    if num_labels == 2:
        temp, train_preds = get_supervised_model_prediction_EM(
            detector, data['train']['text'], data['train']['label'], batch_size, DEVICE, pos_bit)
        _, test_preds = get_supervised_model_prediction_EM(
            detector, data['test']['text'], data['test']['label'], batch_size, DEVICE, pos_bit)
        import pickle as pkl
        maps = {x: y for x, y in zip(temp, train_preds)}
        with open('datasets/%s_%s_%s_%d_%d.pkl' % (
                args.dataset, args.detectLLM, args.method, args.sentence_length, args.iter), 'wb') as file:
            pkl.dump(maps, file)

    else:
        train_preds = get_supervised_model_prediction_multi_classes(
            detector, detector.tokenizer, train_text, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction_multi_classes(
            detector, detector.tokenizer, test_text, batch_size, DEVICE, pos_bit)

    predictions = {
        # 'train': train_preds,
        'test': test_preds,
    }

    y_test_pred_prob = test_preds
    y_test_pred = [round(_) for _ in y_test_pred_prob]
    y_test = test_label

    # train_res = cal_metrics(y_train, y_train_pred, y_train_pred_prob)
    test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
    acc_train, precision_train, recall_train, f1_train, auc_train = 0, 0, 0, 0, 0
    acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test = test_res
    print(
        f"{model} acc_test: {acc_test}, auc_test: {auc_test}, tpr1_test: {tpr1_test}, tpr2_test: {tpr2_test}, tpr3_test: {tpr3_test}")

    # free GPU memory
    del detector
    with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'general': {
            'acc_train': acc_train,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'auc_train': auc_train,
            'acc_test': acc_test,
            'tpr1_test': tpr1_test,
            'tpr2_test': tpr2_test,
            'tpr3_test': tpr3_test,
            'auc_test': auc_test,
        }
    }


def find_best_threshold(predicted_probs, true_labels):
    predicted_probs = np.array(predicted_probs)
    true_labels = np.array(true_labels)
    data_label_0 = np.where(true_labels == 0)[0]
    data_label_1 = np.where(true_labels == 1)[0]

    pred_0 = predicted_probs[data_label_0]
    pred_1 = predicted_probs[data_label_1]

    th_0 = data_label_0[np.argsort(pred_0)[:len(pred_0) // 2]]
    th_1 = data_label_1[np.argsort(-pred_1)[:len(pred_1) // 2]]

    filter_labels = np.zeros_like(true_labels)
    filter_labels[th_0] = 1
    filter_labels[th_1] = 1

    return filter_labels


def get_supervised_model_prediction(
        model,
        tokenizer,
        data,
        batch_size,
        DEVICE,
        pos_bit=0):
    with torch.no_grad():
        # get predictions for real
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            if (args.enhance):
                batch_data = filter(model, tokenizer, batch_data, DEVICE, pos_bit)
            else:
                batch_data = batch_data
            batch_data = tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                # max_length=512,
                return_tensors="pt").to(DEVICE)
            preds.extend(model(**batch_data).logits.softmax(-1)
                         [:, pos_bit].tolist())
    return preds


def get_supervised_model_prediction_EM(
        model,
        x,
        y,
        batch_size,
        DEVICE,
        pos_bit=0,
        pre_train=False):
    train_dataset = my_dataset(x, y)
    with torch.no_grad():
        # get predictions for real
        preds = []
        data = []

        loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        pbar = tqdm(loader, "evaluation")
        for (batch_x, batch_y) in pbar:
            if (args.enhance):
                batch_data = filter(model, model.tokenizer, batch_x, DEVICE, pos_bit)
                data += batch_x
            else:
                batch_data = batch_x
                data += batch_x
            if (pre_train):
                preds.extend(model.pretrain_forward(batch_data)[:, pos_bit].tolist())
            else:
                preds.extend(model(batch_data)[:, pos_bit].tolist())
    return data, preds


def get_supervised_model_prediction_EM_para(
        model,
        x,
        y,
        batch_size,
        DEVICE,
        pos_bit=0,
        pre_train=False):
    train_dataset = my_dataset(x, y)
    with torch.no_grad():
        # get predictions for real
        preds = []
        labels = []

        loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        pbar = tqdm(loader, "evaluation")
        for (batch_x, batch_y) in pbar:
            batch_data = batch_x
            try:
                preds_list, labels_list = model.para_forward(batch_data, batch_y)
                preds.extend(preds_list)
                labels.extend(labels_list)
            except:
                pass
    return preds, labels


def get_supervised_model_prediction_multi_classes(
        model, tokenizer, data, batch_size, DEVICE, pos_bit=0):
    with torch.no_grad():
        # get predictions for real
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            batch_data = tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt").to(DEVICE)
            preds.extend(torch.argmax(
                model(**batch_data).logits, dim=1).tolist())
    return preds


def fine_tune_model(
        model,
        tokenizer,
        data,
        batch_size,
        DEVICE,
        pos_bit=1,
        num_labels=2,
        epochs=3,
        save_path=""):
    # https://huggingface.co/transformers/v3.2.0/custom_datasets.html

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    print(pos_bit)

    if pos_bit == 0 and num_labels == 2:
        train_label = [1 if _ == 0 else 0 for _ in train_label]
        test_label = [1 if _ == 0 else 0 for _ in test_label]

    train_encodings = tokenizer(train_text, truncation=True, padding=True)
    test_encodings = tokenizer(test_text, truncation=True, padding=True)
    train_dataset = CustomDataset(train_encodings, train_label)
    test_dataset = CustomDataset(test_encodings, test_label)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6)

    best_acc = -1
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
        model.eval()
        acc_train, auc_train, tpr1_train, tpr2_train, tpr3_train, acc_test, acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test = eval_data(
            data, model, batch_size, DEVICE, pos_bit,
            tokenizer=tokenizer, flag=False)
        if (auc_train >= best_acc):
            torch.save(model.state_dict(), '%s/model.pth' % save_path)
            best_acc = auc_train


def eval_data(data, detector, batch_size, DEVICE, pos_bit, tokenizer=None, flag=True, pre_train=False):
    train_text = data['val']['text']
    train_label = data['val']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    if (flag):
        _, train_preds = get_supervised_model_prediction_EM(
            detector, train_text, train_label, batch_size, DEVICE, pos_bit)
        _, test_preds = get_supervised_model_prediction_EM(
            detector, test_text, test_label, batch_size, DEVICE, pos_bit, pre_train)
    else:
        train_preds = get_supervised_model_prediction(
            detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction(
            detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)

    y_train_pred_prob = train_preds
    y_train_pred = [round(_) for _ in y_train_pred_prob]
    y_train = train_label

    y_test_pred_prob = test_preds
    y_test_pred = [round(_) for _ in y_test_pred_prob]
    y_test = test_label

    train_res = cal_metrics(y_train, y_train_pred, y_train_pred_prob)
    test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
    acc_train, auc_train, tpr1_train, tpr2_train, tpr3_train = train_res
    acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test = test_res
    print(
        f"acc_train: {acc_train}, auc_train: {auc_train}, tpr1_train: {tpr1_train}, tpr2_train: {tpr2_train}, tpr3_train: {tpr3_train}")
    print(
        f"acc_test: {acc_test}, auc_test: {auc_test}, tpr1_test: {tpr1_test}, tpr2_test: {tpr2_test}, tpr3_test: {tpr3_test}")
    return acc_train, auc_train, tpr1_train, tpr2_train, tpr3_train, acc_test, acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test


def eval_data_para(data, detector, batch_size, DEVICE, pos_bit, tokenizer=None, flag=True, pre_train=False):
    train_text = data['val']['text']
    train_label = data['val']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    train_preds, train_label = get_supervised_model_prediction_EM_para(
        detector, train_text, train_label, batch_size, DEVICE, pos_bit)
    test_preds, test_label = get_supervised_model_prediction_EM_para(
        detector, test_text, test_label, batch_size, DEVICE, pos_bit, pre_train)

    y_train_pred_prob = train_preds
    y_train_pred = [round(_) for _ in y_train_pred_prob]
    y_train = train_label

    y_test_pred_prob = test_preds
    y_test_pred = [round(_) for _ in y_test_pred_prob]
    y_test = test_label

    try:
        train_res = cal_metrics(y_train, y_train_pred, y_train_pred_prob)
        test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
        acc_train, auc_train, tpr1_train, tpr2_train, tpr3_train = train_res
        acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test = test_res
        print(
            f"acc_train: {acc_train}, auc_train: {auc_train}, tpr1_train: {tpr1_train}, tpr2_train: {tpr2_train}, tpr3_train: {tpr3_train}")
        print(
            f"acc_test: {acc_test}, auc_test: {auc_test}, tpr1_test: {tpr1_test}, tpr2_test: {tpr2_test}, tpr3_test: {tpr3_test}")
        return acc_train, auc_train, tpr1_train, tpr2_train, tpr3_train, acc_test, acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test
    except:
        return None


def filter(model, tokenizer, x, DEVICE, pos_bit):
    x_list = []
    with torch.no_grad():
        for i in range(len(x)):
            cur_x_split = split(x[i])

            batch_data = tokenizer(
                cur_x_split,
                padding=True,
                truncation=True,
                return_tensors="pt").to(DEVICE)
            if (args.method == 'MPU'):
                sub_pred = model(cur_x_split).cpu().numpy()[:, pos_bit]
            else:
                sub_pred = model(**batch_data).logits.softmax(-1).cpu().numpy()[:, pos_bit]

            index = np.argsort(sub_pred)

            filter_list = []
            for ind in index:
                if (sub_pred[ind] < args.conf_threshold):
                    filter_list.append(ind)
            filter_list = filter_list[:int(len(cur_x_split) * args.filter_threshold)]
            index = []
            for j in range(len(cur_x_split)):
                if (j in filter_list):
                    continue
                else:
                    index.append(j)
            # index = (index * 10)[:len(cur_x_split)]
            cur_x = ""
            for j in range(len(index)):
                if (j == 0):
                    cur_x += cur_x_split[index[j]]
                else:
                    if (index[j] == index[j - 1] + 1):
                        cur_x += " " + cur_x_split[index[j]]
                    else:
                        # cur_x += " " + cur_x_split[index[j]]
                        cur_x += " </s> " + cur_x_split[index[j]]
            cur_x += ""
            x_list.append(cur_x)

    return x_list


def split(x):
    cur_x_split = sent_tokenize(x)
    sentence_list = cur_x_split

    # sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!) ')
    # cur_x_split = sentence_endings.split(x)
    # # cur_x_split = [s.strip(" ") for s in cur_x_split]

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


def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)

    return log_likelihood_x.cpu().data.numpy(), log_likelihood_x_tilde.cpu().data.numpy()


def get_text_crit(text, args, model_config):
    tokenized = model_config["scoring_tokenizer"](text, return_tensors="pt",
                                                  return_token_type_ids=False, max_length=1024).to(args.DEVICE)
    labels = tokenized.input_ids[:, 1:]
    with torch.no_grad():
        logits_score = model_config["scoring_model"](**tokenized).logits[:, :-1]
        # if args.reference_model == args.scoring_model:
        #     logits_ref = logits_score
        # else:
        tokenized = model_config["reference_tokenizer"](text, return_tensors="pt",
                                                        return_token_type_ids=False, max_length=1024).to(args.DEVICE)
        assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
        logits_ref = model_config["reference_model"](**tokenized).logits[:, :-1]
        tlog_likelihood_x, log_likelihood_x_tilde = get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)

    return tlog_likelihood_x, log_likelihood_x_tilde


def run_fast_experiments(data, base_model, base_tokenizer, score_model, score_tokenizer):
    model_config = {
        "reference_tokenizer": base_tokenizer,
        "reference_model": base_model,
        "scoring_tokenizer": score_tokenizer,
        "scoring_model": score_model,
    }

    for nn in ['train', 'test', 'val']:
        test_text = data[nn]['text']
        test_label = data[nn]['label']

        likelihood = []
        likelihood_tilde = []
        for text in tqdm(test_text):
            log_likelihood_x, log_likelihood_x_tilde = get_text_crit(text, args, model_config)
            if (len(log_likelihood_x[0]) == 1023):
                likelihood.append(log_likelihood_x)
                likelihood_tilde.append(log_likelihood_x_tilde)
            else:
                likelihood_temp = np.zeros((1, 1023, 1))
                likelihood_tilde_temp = np.zeros((1, 1023, 10))
                likelihood_temp[:, :len(log_likelihood_x[0])] = log_likelihood_x
                likelihood_tilde_temp[:, :len(log_likelihood_x[0])] = log_likelihood_x_tilde
                likelihood.append(likelihood_temp)
                likelihood_tilde.append(likelihood_tilde_temp)

        likelihood = np.concatenate(likelihood, axis=0)
        likelihood_tilde = np.concatenate(likelihood_tilde, axis=0)

        np.save("datasets/embedding/x_likelihood_%s_%s_%s_%s_%d_%d.npy" % (nn,
                                                                    args.dataset, args.detectLLM, args.method,
                                                                    args.sentence_length, args.iter), likelihood)
        np.save("datasets/embedding/x_likelihood_tilde_%s_%s_%s_%s_%d_%d.npy" % (nn,
                                                                     args.dataset, args.detectLLM, args.method,
                                                                     args.sentence_length, args.iter), likelihood_tilde)
        np.save("datasets/embedding/y_%s_%s_%s_%s_%d_%d.npy" % (nn,
                                                                args.dataset, args.detectLLM, args.method,
                                                                args.sentence_length, args.iter), test_label)
    exit(0)
