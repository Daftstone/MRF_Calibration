from parse import args
import datetime
import os
import json
import dataset_loader_sentence
from methods.utils import load_base_model, load_base_model_and_tokenizer
from methods.supervised import run_supervised_experiment_New
from methods.detectgpt import run_perturbation_experiments
from methods.dnagpt import run_dna_experiment
from methods.supervised import run_fast_experiments
from methods.gptzero import run_gptzero_experiment
from methods.metric_based import get_prob, get_loss, get_rank_detail, get_entropy_detail, get_ll, get_rank, get_entropy, \
    get_rank_GLTR, \
    run_threshold_experiment, \
    run_GLTR_experiment

if __name__ == '__main__':

    if (args.sentence_length == -1):
        args.batch_size = 16
        args.paragraph_num = 32

    import random
    import torch

    random.seed(0)
    seeds = [random.randint(0, 100000000) for _ in range(100)]

    random.seed(seeds[args.iter])
    torch.manual_seed(seeds[args.iter])
    torch.cuda.manual_seed(seeds[args.iter])
    torch.cuda.manual_seed_all(seeds[args.iter])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    DEVICE = args.DEVICE

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    print(f'Loading dataset {args.dataset}...')
    data = dataset_loader_sentence.load(args.dataset, detectLLM=args.detectLLM)

    base_model_name = args.base_model_name.replace('/', '_')
    SAVE_PATH = f"update_results/{base_model_name}-{args.mask_filling_model_name}/{args.dataset}-{args.detectLLM}"
    if (len(args.load_path) > 0):
        temp = args.load_path.split("-")
        if (len(temp) == 2):
            train_model = temp[1]
        else:
            train_model = "-".join(temp[1:])
    else:
        train_model = args.detectLLM
    args.train_model = train_model
    if (len(args.load_path) > 0):
        args.load_path_criterion = f"update_results/{base_model_name}-{args.mask_filling_model_name}/{args.load_path}/{train_model}_{args.method}_benchmark_results_{args.sentence_num}_{args.sentence_length}_{args.adversarial}_{args.reg}_{args.iter}.pkl"
    if (len(args.load_path) > 0):
        args.load_path = f"update_results/{base_model_name}-{args.mask_filling_model_name}/{args.load_path}/{args.method}"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if not os.path.exists(SAVE_PATH + '/%s' % args.method):
        os.makedirs(SAVE_PATH + '/%s' % args.method)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_PATH)}")

    # write args to file
    with open(os.path.join(SAVE_PATH, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    mask_filling_model_name = args.mask_filling_model_name
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # get generative model
    if (args.method in ['Log_Likelihood', 'Log_Rank', 'Entropy', 'FastGPT', 'DetectGPT', 'DNAGPT']):
        base_model, base_tokenizer = load_base_model_and_tokenizer(
            args.base_model_name, cache_dir)
        load_base_model(base_model, DEVICE)


    def loss_criterion(text):
        return get_loss(
            text, base_model, base_tokenizer, DEVICE)



    def logrank_detail_criterion(text):
        return get_rank_detail(
            text, base_model, base_tokenizer, DEVICE, log=True)


    def entropy_detail_criterion(text):
        return get_entropy_detail(
            text, base_model, base_tokenizer, DEVICE)


    outputs = []
    # outputs.append(run_threshold_experiment(data, phd_criterion, "phd"))

    if args.method == "Log_Rank":
        run_threshold_experiment(data, logrank_detail_criterion, "Log_Rank", args.load_path_criterion)
    elif args.method == "Entropy":
        run_threshold_experiment(data, entropy_detail_criterion, "Entropy", args.load_path_criterion)
    elif args.method == "Log_Likelihood":
        run_threshold_experiment(data, loss_criterion, "Log_Likelihood", args.load_path_criterion)
    elif args.method == "FastGPT":
        score_model, score_tokenizer = load_base_model_and_tokenizer(
            'gpt2-xl', cache_dir)
        load_base_model(score_model, DEVICE)
        run_fast_experiments(
            data, base_model, base_tokenizer, score_model, score_tokenizer)
    elif args.method == "DetectGPT":
        run_perturbation_experiments(
            args, data, base_model, base_tokenizer, args.load_path_criterion, method="DetectGPT")
    elif args.method == "DNAGPT":
        run_dna_experiment(
            args, data, method="DetectGPT")
