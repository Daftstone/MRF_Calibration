from parse import args
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from methods.utils import timeit, get_clf_results
from methods.IntrinsicDim import PHD
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import pickle as pkl


# # Under development
# def get_phd(text, base_model, base_tokenizer, DEVICE):
#     # default setting
#     MIN_SUBSAMPLE = 40
#     INTERMEDIATE_POINTS = 7
#     alpha=1.0
#     solver = PHD(alpha=alpha, metric='euclidean', n_points=9)

#     text = text[:200]
#     inputs = base_tokenizer(text, truncation=True, max_length=1024, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         outp = base_model(**inputs)

#     # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
#     mx_points = inputs['input_ids'].shape[1] - 2
#     mn_points = MIN_SUBSAMPLE
#     step = ( mx_points - mn_points ) // INTERMEDIATE_POINTS

#     t1 = time.time()
#     res = solver.fit_transform(outp[0][0].cpu().numpy()[1:-1],  min_points=mn_points, max_points=mx_points - step, point_jump=step)
#     print(time.time() - t1, "Seconds")
#     return res

def weighted_smoothing(arr):
    # 边缘填充（复制边界值）
    padded = np.pad(arr, pad_width=1, mode='edge')
    # 定义滤波核
    kernel = np.array([0.25, 0.5, 0.25])
    # 执行卷积（valid模式确保输出长度与输入一致）
    return np.convolve(padded, kernel, mode='valid')


def get_loss(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids

        logits = base_model(**tokenized).logits

        labels = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()
        logits = logits.view(-1, base_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(logits.device)
        loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction='none')

        loss = -loss.cpu().numpy()
        return loss


def get_prob(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        prob = base_model(**tokenized).logits.softmax(-1)[0].cpu().numpy()
        all_prob = prob[np.arange(prob.shape[0]), labels.cpu().numpy()]
        return all_prob[0]


# get the average rank of each observed token sorted by model likelihood
def get_rank_detail(text, base_model, base_tokenizer, DEVICE, log=False):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                   == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[
                   1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(
            timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return -np.array(ranks.float().cpu().numpy())


# get average entropy of each token in the text
def get_entropy_detail(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=1024,
            return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return neg_entropy.sum(-1).cpu().numpy()[0]


def get_ll(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L1317


def get_lls(texts, base_model, base_tokenizer, DEVICE):
    return [get_loss(_, base_model, base_tokenizer, DEVICE) for _ in texts]


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, base_model, base_tokenizer, DEVICE, log=False):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                   == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[
                   1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(
            timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


def get_ranks(texts, base_model, base_tokenizer, DEVICE, log=False):
    return [get_rank(_, base_model, base_tokenizer, DEVICE, log)
            for _ in texts]


def get_rank_GLTR(text, base_model, base_tokenizer, DEVICE, log=False):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=1024,
            return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                   == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[
                   1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(
            timesteps.device)).all(), "Expected one match per timestep"
        ranks = ranks.float()
        res = np.array([0.0, 0.0, 0.0, 0.0])
        for i in range(len(ranks)):
            if ranks[i] < 10:
                res[0] += 1
            elif ranks[i] < 100:
                res[1] += 1
            elif ranks[i] < 1000:
                res[2] += 1
            else:
                res[3] += 1
        if res.sum() > 0:
            res = res / res.sum()

        return res


# get average entropy of each token in the text
def get_entropy(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


@timeit
def run_threshold_experiment(data, criterion_fn, name, load_path):
    torch.manual_seed(0)
    np.random.seed(0)

    for nn in ['train', 'test', 'val']:
        train_text = data[nn]['text']
        train_label = data[nn]['label']
        train_criterion = [
            criterion_fn(
                train_text[idx]) for idx in tqdm(
                range(
                    len(train_text)),
                desc="%s criterion" % nn)]
        if (name in ['Log_Likelihood', 'Log_Rank', 'Entropy']):
            max_length = max(len(v) for v in train_criterion)
            padded_train_criterion = np.zeros((len(train_criterion), max_length), dtype=np.float32)
            for i, v in enumerate(train_criterion):
                padded_train_criterion[i, :len(v)] = v
            x_train = padded_train_criterion
        else:
            x_train = np.array(train_criterion)

        y_train = np.array(train_label)

        select_train_index = ~np.isnan(x_train).any(axis=1)
        x_train = x_train[select_train_index]
        y_train = y_train[select_train_index]

        np.save("datasets/embedding/x_%s_%s_%s_%s_%d_%d.npy" % (nn,
                                                                args.dataset, args.detectLLM, args.method,
                                                                args.sentence_length, args.iter), x_train)
        np.save("datasets/embedding/y_%s_%s_%s_%s_%d_%d.npy" % (nn,
                                                                args.dataset, args.detectLLM, args.method,
                                                                args.sentence_length, args.iter), y_train)
    exit(0)


@timeit
def run_threshold_experiment_multiple_test_length(
        clf, data, criterion_fn, name, lengths=[
            10, 20, 50, 100, 200, 500, -1]):
    torch.manual_seed(0)
    np.random.seed(0)
    res = {}
    from methods.utils import cut_length, cal_metrics
    for length in lengths:
        test_text = data['test']['text']
        test_label = data['test']['label']
        test_criterion = [
            criterion_fn(
                cut_length(
                    test_text[idx],
                    length)) for idx in tqdm(
                range(
                    len(test_text)),
                desc="Test criterion")]
        x_test = np.array(test_criterion)
        y_test = np.array(test_label)

        # remove nan values
        select_test_index = ~np.isnan(x_test)
        x_test = x_test[select_test_index]
        y_test = y_test[select_test_index]
        x_test = np.expand_dims(x_test, axis=-1)

        y_test_pred = clf.predict(x_test)
        y_test_pred_prob = clf.predict_proba(x_test)
        y_test_pred_prob = [_[1] for _ in y_test_pred_prob]
        acc_test, precision_test, recall_test, f1_test, auc_test = cal_metrics(
            y_test, y_test_pred, y_test_pred_prob)
        test_res = acc_test, precision_test, recall_test, f1_test, auc_test

        print(
            f"{name} {length} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")
        res[length] = test_res

    return res


@timeit
def run_GLTR_experiment(data, criterion_fn, name, load_path):
    torch.manual_seed(0)
    np.random.seed(0)

    x_train = np.load(
        "datasets/embedding/x_train_%s_%s_%s_%d.npy" % (args.dataset, args.detectLLM, args.method, args.iter))
    y_train = np.load(
        "datasets/embedding/y_train_%s_%s_%s_%d.npy" % (args.dataset, args.detectLLM, args.method, args.iter))
    x_test = np.load(
        "datasets/embedding/x_test_%s_%s_%s_%d.npy" % (args.dataset, args.detectLLM, args.method, args.iter))
    y_test = np.load(
        "datasets/embedding/y_test_%s_%s_%s_%d.npy" % (args.dataset, args.detectLLM, args.method, args.iter))

    # train_text = data['train']['text']
    # train_label = data['train']['label']
    # train_criterion = [criterion_fn(train_text[idx])
    #                    for idx in range(len(train_text))]
    # x_train = np.array(train_criterion)
    # y_train = train_label
    #
    # test_text = data['test']['text']
    # test_label = data['test']['label']
    # test_criterion = [criterion_fn(test_text[idx])
    #                   for idx in range(len(test_text))]
    # x_test = np.array(test_criterion)
    # y_test = test_label
    #
    # np.save("datasets/embedding/x_train_%s_%s_%s_%d.npy" % (args.dataset, args.detectLLM, args.method, args.iter),
    #         x_train)
    # np.save("datasets/embedding/y_train_%s_%s_%s_%d.npy" % (args.dataset, args.detectLLM, args.method, args.iter),
    #         y_train)
    # np.save("datasets/embedding/x_test_%s_%s_%s_%d.npy" % (args.dataset, args.detectLLM, args.method, args.iter),
    #         x_test)
    # np.save("datasets/embedding/y_test_%s_%s_%s_%d.npy" % (args.dataset, args.detectLLM, args.method, args.iter),
    #         y_test)
    # exit(0)

    clf, train_res, test_res = get_clf_results(
        x_train, y_train, x_test, y_test, load_path)

    acc_train, precision_train, recall_train, f1_train, auc_train = train_res
    acc_test, precision_test, recall_test, f1_test, auc_test = test_res

    print(
        f"{name} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
    print(
        f"{name} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

    return {
        'name': f'{name}_threshold',
        'predictions': {'train': x_train, 'test': x_test},
        'general': {
            'acc_train': acc_train,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'auc_train': auc_train,
            'acc_test': acc_test,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'f1_test': f1_test,
            'auc_test': auc_test,
        },
        'clf': clf
    }


@timeit
def run_GLTR_experiment_multiple_test_length(
        clf, data, criterion_fn, name, lengths=[
            10, 20, 50, 100, 200, 500, -1]):
    torch.manual_seed(0)
    np.random.seed(0)

    res = {}
    from methods.utils import cut_length, cal_metrics
    for length in lengths:
        test_text = data['test']['text']
        test_label = data['test']['label']
        test_criterion = [
            criterion_fn(
                cut_length(
                    test_text[idx],
                    length)) for idx in tqdm(
                range(
                    len(test_text)),
                desc="Test criterion")]
        x_test = np.array(test_criterion)
        y_test = np.array(test_label)

        y_test_pred = clf.predict(x_test)
        y_test_pred_prob = clf.predict_proba(x_test)
        y_test_pred_prob = [_[1] for _ in y_test_pred_prob]
        acc_test, precision_test, recall_test, f1_test, auc_test = cal_metrics(
            y_test, y_test_pred, y_test_pred_prob)
        test_res = acc_test, precision_test, recall_test, f1_test, auc_test

        print(
            f"{name} {length} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")
        res[length] = test_res

    return res


def filter(clf, x, criterion_fn):
    x_list = []
    with torch.no_grad():
        for i in range(len(x)):
            cur_x_split = split(x[i])

            train_criterion = [criterion_fn(cur_x_split[idx]) for idx in range(len(cur_x_split))]
            x_train = np.array(train_criterion)

            x_train = np.expand_dims(x_train, axis=-1)

            y_train_pred_prob = clf.predict_proba(x_train)
            sub_pred = [_[1] for _ in y_train_pred_prob]

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
            index = (index * 10)[:len(cur_x_split)]
            cur_x = ""
            for j in range(len(index)):
                if (j == 0):
                    cur_x += cur_x_split[index[j]]
                else:
                    if (index[j] == index[j - 1] + 1):
                        cur_x += " " + cur_x_split[index[j]]
                    else:
                        cur_x += " </s> " + cur_x_split[index[j]]
            cur_x += ""
            x_list.append(cur_x)

    return x_list


def split(x):
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
