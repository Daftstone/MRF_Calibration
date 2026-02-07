import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="Essay")
parser.add_argument('--method', type=str, default="EM")
parser.add_argument('--gpu', type=str, default="cuda:0")
parser.add_argument('--sentence_num', type=int, default=3)
parser.add_argument('--sentence_length', type=int, default=-1)
parser.add_argument('--adversarial', type=int, default=1)
parser.add_argument('--reg', type=float, default=0.1)

args = parser.parse_args()

for method in ["Log_Likelihood", "Log_Rank", "Entropy", "DetectGPT", "FastGPT", "DNAGPT"]:
    args.method = method
    for dataset in ['Essay', 'DetectRL', 'Reuters', 'Mix', "dipper", 'polish', 'Cross']:
        args.dataset = dataset
        if (
                args.dataset == 'DetectRL' or args.dataset == 'Mix' or args.dataset == 'dipper' or args.dataset == 'polish'):
            llms = ['ChatGPT', 'Google-PaLM', 'Llama-2-70b']
        elif (args.dataset == 'Cross'):
            llms = ['writing_prompt', 'yelp_review']
            llms = ['arxiv', 'writing_prompt', 'xsum', 'yelp_review']
        else:
            llms = ['ChatGPT-turbo', 'Claude', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4All']

        for llm in llms:
            os.system(
                "python test.py --DEVICE %s --method %s --dataset %s --train_model %s --sentence_num %d --sentence_length %d --adversarial %d --reg %s| tee logs/temp_%s_%s_%s_%d_%d_%d_%s.txt" % (
                    args.gpu, args.method, args.dataset, llm, args.sentence_num, args.sentence_length, args.adversarial,
                    args.reg, args.dataset,
                    llm, args.method, args.sentence_num, args.sentence_length, args.adversarial, args.reg))
