from parse import args
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from methods.utils import cal_metrics
import os
import pickle as pkl
import random

device = args.DEVICE


def get_data(feature1, feature2):
    data = np.zeros((len(feature1), 11, 1024), dtype=np.float32)
    for k, (data1, data2) in enumerate(zip(feature1, feature2)):
        for i in range(10):
            temp = data1[i]
            data[k, i, :len(temp)] = temp
        temp = data2
        data[k, 10, :len(temp)] = temp
    return data


def get_mask(x):
    mask = np.zeros_like(x)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            idx = np.where(x[i, j] == 0)[0]
            if (len(idx) == 0 or np.any(x[i, j][idx[-1]:] != 0)):
                mask[i, j] = 1
            else:
                for iidx in idx:
                    if (np.all(x[i, j][iidx:] == 0)):
                        mask[i, j, :iidx] = 1
                        break
    return mask


def evaluate(preds, train_y):
    y_test_pred_prob = preds
    y_test_pred = [round(_) for _ in y_test_pred_prob]
    y_test = train_y
    test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
    acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test = test_res
    print(
        f"acc_test: {acc_test}, auc_test: {auc_test}, tpr1_test: {tpr1_test}, tpr2_test: {tpr2_test}, tpr3_test: {tpr3_test}")
    return acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test


def cur_evaluation(model, cur_dataset, data_loader):
    pred_list = []
    label_list = []
    pbar = tqdm(data_loader, desc=f"Evaluation: ")
    with torch.no_grad():
        for x, m, y in pbar:
            x = x.to(device)
            m = m.to(device)
            adj_in = torch.tensor(train_dataset.adj_in).to(device).to_sparse()
            outputs = model(x, adj_in, m)
            outputs = outputs.cpu().numpy()
            pred_list.append(outputs)
            label_list.append(y)

    preds = np.concatenate(pred_list)
    labels = np.concatenate(label_list)
    acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test = evaluate(preds, labels)
    return acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test


class my_dataset(Dataset):
    def __init__(self, data, labels, filter_pos):
        self.labels = labels
        self.mask = get_mask(data)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        self.data = np.concatenate([1. - data[:, :, :, None], data[:, :, :, None]], axis=-1)
        self.adj_in = np.zeros((len(data[0, 0]), len(data[0, 0])), dtype=np.float32)
        if (filter_pos > 0):
            for i in range(1, len(self.adj_in)):
                self.adj_in[i, i - 1] = 1 / (1 + np.exp(-(i - 1 - filter_pos)))
                self.adj_in[i, i] = 1
            for i in range(0, len(self.adj_in) - 1):
                self.adj_in[i, i + 1] = 1 / (1 + np.exp(-(i + 1 - filter_pos)))
                self.adj_in[i, i] = 1
        else:
            for i in range(1, len(self.adj_in)):
                self.adj_in[i, i - 1] = 1
                self.adj_in[i, i] = 1
            for i in range(0, len(self.adj_in) - 1):
                self.adj_in[i, i + 1] = 1
                self.adj_in[i, i] = 1

    def __getitem__(self, index):
        cur_data = self.data[index]
        cur_label = self.labels[index]
        cur_mask = self.mask[index]
        return cur_data, cur_mask, cur_label

    def __len__(self):
        return len(self.labels)


class GCN(nn.Module):
    def __init__(self, device, t, filter_pos):
        super(GCN, self).__init__()
        self.device = device
        self.w1 = nn.Parameter(torch.randn(1) * 0.001)
        self.w11 = nn.Parameter(torch.randn(1) * 0.001)
        self.w_prime1 = nn.Parameter(torch.randn(1) * 0.001)
        self.w_prime11 = nn.Parameter(torch.randn(1) * 0.001)
        self.w2 = nn.Parameter(torch.randn(1) * 0.001)
        self.w22 = nn.Parameter(torch.randn(1) * 0.001)
        self.w_prime2 = nn.Parameter(torch.randn(1) * 0.001)
        self.w_prime22 = nn.Parameter(torch.randn(1) * 0.001)
        self.w_final = nn.Parameter(torch.randn(1) * 0.0001)
        self.b_final = nn.Parameter(torch.randn(1) * 0.)
        self.t = t
        self.filter_pos = filter_pos
        self.weights = torch.tensor([1 / (1 + np.exp(-(i - self.filter_pos))) for i in range(1024)],
                                    dtype=torch.float32).to(
            device)[None, None, :]
        self.to(device)

    def forward(self, x, adj_in, m):
        diag = torch.diagflat(torch.stack([-torch.abs(self.w1), -torch.abs(self.w11)]))
        off_diag = torch.fliplr(torch.diagflat(torch.stack([torch.abs(self.w_prime1), torch.abs(self.w_prime11)])))
        self.w_in = diag + off_diag
        diag = torch.diagflat(torch.stack([-torch.abs(self.w2), -torch.abs(self.w22)]))
        off_diag = torch.fliplr(torch.diagflat(torch.stack([torch.abs(self.w_prime2), torch.abs(self.w_prime22)])))
        self.w_out = diag + off_diag

        H = x[:, -1]

        for i in range(self.t):
            x2 = torch.matmul(
                torch.sparse.mm(adj_in, H.permute(1, 2, 0).reshape((1024, -1))).reshape((1024, 2, -1)).permute(
                    2, 0,
                    1),
                self.w_in)
            H = torch.softmax(torch.log(H + 1e-10) - x2, dim=-1)

        x = x[:, :, :, 1]

        if (self.filter_pos > 0):
            std = self.get_std(x[:, :10], m[:, :10])
            mean = self.get_mean(x[:, :10], m[:, :10])
            mean_cur = self.get_mean(H[:, None, :, 1], m[:, -1:])
        else:
            std = self.get_std_norm(x[:, :10], m[:, :10])
            mean = self.get_mean_norm(x[:, :10], m[:, :10])
            mean_cur = self.get_mean_norm(H[:, None, :, 1], m[:, -1:])

        x = (mean_cur - mean) / (std + 1e-6)
        x = torch.sigmoid(x * self.w_final + self.b_final)
        return x

    def get_std(self, A, M):
        masked_A = A * M
        output = torch.sum(masked_A * self.weights, dim=2) / (torch.sum(M * self.weights, dim=2) + 1e-10)
        output = torch.std(output, dim=1)
        return output

    def get_mean(self, A, M):
        masked_A = A * M
        output = torch.sum(masked_A * self.weights, dim=2) / (torch.sum(M * self.weights, dim=2) + 1e-10)
        output = torch.mean(output, dim=1)
        return output

    def get_std_norm(self, A, M):
        masked_A = A * M
        output = torch.sum(masked_A * self.weights, dim=2) / (torch.sum(M * self.weights, dim=2) + 1e-10)
        output = torch.std(output, dim=1)
        return output

    def get_mean_norm(self, A, M):
        masked_A = A * M
        output = torch.sum(masked_A * self.weights, dim=2) / (torch.sum(M * self.weights, dim=2) + 1e-10)
        output = torch.mean(output, dim=1)
        return output


for filter_pos, t in zip([0, 30], [0, 10]):
    for method in ['DetectGPT']:
        # for dataset in ['Essay', 'Reuters', 'DetectRL', 'Mix', 'back_translation', 'dipper', 'polish']:
        for dataset in [args.dataset]:
            if (dataset in ['Mix', "dipper", 'polish', 'DetectRL']):
                llms = ['ChatGPT', 'Claude-instant', 'Google-PaLM', 'Llama-2-70b']
                LLMs = ['ChatGPT', 'Claude-instant', 'Google-PaLM', 'Llama-2-70b']
                ref_dataset = 'DetectRL'
                ttest_dataset = dataset
            elif (dataset in ['Mix1']):
                llms = ['ChatGPT', 'Claude-instant', 'Google-PaLM', 'Llama-2-70b']
                LLMs = ['ChatGPT', 'Claude-instant', 'Google-PaLM', 'Llama-2-70b']
                ref_dataset = 'Mix'
                ttest_dataset = 'Mix'
            elif (dataset in ['Cross']):
                llms = ['arxiv', 'writing_prompt', 'xsum', 'yelp_review']
                LLMs = ['arxiv', 'writing_prompt', 'xsum', 'yelp_review']
                ref_dataset = 'Cross'
                ttest_dataset = dataset
            else:
                llms = ['ChatGPT', 'ChatGPT-turbo', 'ChatGLM', 'Claude', 'Dolly', 'GPT4All', 'StableLM']
                LLMs = ['ChatGPT', 'ChatGPT-turbo', 'ChatGLM', 'Claude', 'Dolly', 'GPT4All', 'StableLM']
                ref_dataset = dataset
                ttest_dataset = dataset

            lr_map = {'Log_Rank_detail': 0.05, 'loss': 0.05, 'Entropy_detail': 0.05, 'DetectGPT': 0.05}

            for LLM in LLMs:
                for iter in range(5):

                    random.seed(iter)
                    torch.manual_seed(iter)
                    torch.cuda.manual_seed(iter)
                    torch.cuda.manual_seed_all(iter)
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True

                    print(LLM, iter)

                    with open(
                            'datasets/embedding/x_train_feature1_%s_%s_%s_-1_%d.pkl' % (
                                    ref_dataset, LLM, method, iter),
                            'rb') as f:
                        feature1 = pkl.load(f)
                    with open(
                            'datasets/embedding/x_train_feature2_%s_%s_%s_-1_%d.pkl' % (
                                    ref_dataset, LLM, method, iter),
                            'rb') as f:
                        feature2 = pkl.load(f)
                    train_x = get_data(feature1, feature2)

                    with open('datasets/embedding/y_train_%s_%s_%s_-1_%d.pkl' % (ref_dataset, LLM, method, iter),
                              'rb') as f:
                        train_y = pkl.load(f)

                    with open(
                            'datasets/embedding/x_val_feature1_%s_%s_%s_-1_%d.pkl' % (
                                    ref_dataset, LLM, method, iter),
                            'rb') as f:
                        feature1 = pkl.load(f)
                    with open(
                            'datasets/embedding/x_val_feature2_%s_%s_%s_-1_%d.pkl' % (
                                    ref_dataset, LLM, method, iter),
                            'rb') as f:
                        feature2 = pkl.load(f)
                    val_x = get_data(feature1, feature2)
                    with open('datasets/embedding/y_val_%s_%s_%s_-1_%d.pkl' % (ref_dataset, LLM, method, iter),
                              'rb') as f:
                        val_y = pkl.load(f)

                    train_dataset = my_dataset(train_x, train_y, filter_pos)
                    train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)

                    val_dataset = my_dataset(val_x, val_y, filter_pos)
                    val_loader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False)

                    # 初始化模型
                    model = GCN(device, t, filter_pos)
                    criterion = nn.BCELoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr_map[method])

                    auc_best = -1
                    adj_in = torch.tensor(train_dataset.adj_in).to(device).to_sparse()
                    for epoch in range(10):
                        pbar = tqdm(train_loader, desc=f"Training: {epoch} epoch")
                        for x, m, y in pbar:
                            x = x.to(device)
                            m = m.to(device)
                            y = y.to(device).float()

                            outputs = model(x, adj_in, m)

                            loss = criterion(outputs, y)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        acc_val, auc_val, tpr1_val, tpr2_val, tpr3_val = cur_evaluation(model, val_dataset,
                                                                                        val_loader)

                        if (auc_val > auc_best):
                            torch.save(model.state_dict(),
                                       f'save_models/%s_%s_%s_%d_%d.pth' % (dataset, method, LLM, filter_pos, t))
                            auc_best = auc_val
                    model.load_state_dict(torch.load(
                        'save_models/%s_%s_%s_%d_%d.pth' % (dataset, method, LLM, filter_pos, t),
                        map_location=model.device))

                    for llm in llms:
                        with open(
                                'datasets/embedding/x_test_feature1_%s_%s_%s_-1_%d.pkl' % (
                                        ttest_dataset, llm, method, iter),
                                'rb') as f:
                            feature1 = pkl.load(f)
                        with open(
                                'datasets/embedding/x_test_feature2_%s_%s_%s_-1_%d.pkl' % (
                                        ttest_dataset, llm, method, iter),
                                'rb') as f:
                            feature2 = pkl.load(f)
                        print('datasets/embedding/x_test_feature1_%s_%s_%s_-1_%d.pkl' % (
                            ttest_dataset, llm, method, iter))
                        test_x = get_data(feature1, feature2)
                        with open(
                                'datasets/embedding/y_test_%s_%s_%s_-1_%d.pkl' % (ttest_dataset, llm, method, iter),
                                'rb') as f:
                            test_y = pkl.load(f)

                        test_dataset = my_dataset(test_x, test_y, filter_pos)
                        test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

                        acc_test, auc_test, tpr1_test, tpr2_test, tpr3_test = cur_evaluation(model, test_dataset,
                                                                                             test_loader)

                        outputs = {
                            'general': {
                                'acc_test': acc_test,
                                'tpr1_test': tpr1_test,
                                'tpr2_test': tpr2_test,
                                'tpr3_test': tpr3_test,
                                'auc_test': auc_test,
                            }
                        }

                        base_model_name = 'gpt2-medium'
                        mask_filling_model_name = 't5-base'
                        detectLLM = LLM
                        SAVE_PATH = f"update_results/{base_model_name}-{mask_filling_model_name}/{dataset}-{detectLLM}"
                        if not os.path.exists(SAVE_PATH):
                            os.makedirs(SAVE_PATH)
                        with open(os.path.join(SAVE_PATH,
                                               f"{llm}_{method}_benchmark_results_{filter_pos}_{t}_{iter}.pkl"),
                                  "wb") as f:
                            pkl.dump(outputs, f)

            for detectLLM in LLMs:
                with open('logs/temp_%s_%s_%s_%d_%d.txt' % (dataset, method, detectLLM, filter_pos, t),
                          'w') as ff:
                    for llm in llms:
                        for iter in range(5):
                            with open(os.path.join(f"update_results/gpt2-medium-t5-base/{dataset}-{detectLLM}",
                                                   f"{llm}_{method}_benchmark_results_{filter_pos}_{t}_{iter}.pkl"),
                                      "rb") as f:
                                output = pkl.load(f)

                                ff.write(
                                    "%s %f %f %f %f %f\n" % (
                                        llm, output['general']['acc_test'], output['general']['auc_test'],
                                        output['general']['tpr1_test'], output['general']['tpr2_test'],
                                        output['general']['tpr3_test']))

                                print(llm, output['general']['acc_test'], output['general']['auc_test'],
                                      output['general']['tpr1_test'], output['general']['tpr2_test'],
                                      output['general']['tpr3_test'])
