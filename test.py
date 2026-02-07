import os
import argparse
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--DEVICE', type=str, default="cuda:0")
parser.add_argument('--method', type=str, default="EM")
parser.add_argument('--dataset', type=str, default="Essay")
parser.add_argument('--train_model', type=str, default="ChatGPT-turbo")
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--reg', type=float, default=0.1)
parser.add_argument('--sentence_num', type=int, default=4)
parser.add_argument('--sentence_length', type=int, default=1)
parser.add_argument('--adversarial', type=int, default=1)

args = parser.parse_args()

if (args.dataset in ['DetectRL', 'Mix', 'dipper', 'polish', 'CMix', 'back_translation', 'CCMix']):
    LLMS = ['ChatGPT', 'Google-PaLM', 'Llama-2-70b']
elif (args.dataset in ['Cross']):
    LLMS = ['arxiv', 'writing_prompt', 'xsum', 'yelp_review']
else:
    LLMS = ['ChatGPT-turbo', 'ChatGLM', 'ChatGPT', 'GPT4All', 'Claude', 'Dolly']

for i in range(5):
    os.system(
        "python benchmark.py --dataset %s --detectLLM %s --method %s --DEVICE %s --finetune --iter %d --reg %s --sentence_num %d --sentence_length %d --adversarial %d" % (
            args.dataset, args.train_model, args.method, args.DEVICE, i, args.reg, args.sentence_num,
            args.sentence_length, args.adversarial))