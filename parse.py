import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="Essay")
parser.add_argument('--detectLLM', type=str, default="ChatGPT")
parser.add_argument('--train_model', type=str, default="ChatGPT")
parser.add_argument('--method', type=str, default="Log-Likelihood")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
parser.add_argument('--mask_filling_model_name',
                    type=str, default="t5-base")
parser.add_argument('--cache_dir', type=str, default="save_models")
parser.add_argument('--DEVICE', type=str, default="cuda:0")

# params for DetectGPT
parser.add_argument('--pct_words_masked', type=float, default=0.3)
parser.add_argument('--span_length', type=int, default=2)
parser.add_argument('--n_perturbation_list', type=str, default="10")
parser.add_argument('--n_perturbation_rounds', type=int, default=1)
parser.add_argument('--chunk_size', type=int, default=20)
parser.add_argument('--n_similarity_samples', type=int, default=20)
parser.add_argument('--int8', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--do_top_k', action='store_true')
parser.add_argument('--top_k', type=int, default=40)
parser.add_argument('--do_top_p', action='store_true')
parser.add_argument('--top_p', type=float, default=0.96)
parser.add_argument('--buffer_size', type=int, default=1)
parser.add_argument('--mask_top_p', type=float, default=1.0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--random_fills', action='store_true')
parser.add_argument('--random_fills_tokens', action='store_true')

# params for GPTZero
parser.add_argument('--gptzero_key', type=str, default="")

parser.add_argument('--finetune', action='store_true')
parser.add_argument('--load_path', type=str, default="")
parser.add_argument('--load_path_criterion', type=str, default="")
parser.add_argument('--enhance', type=int, default=0)

parser.add_argument('--reg', type=float, default=0.1)
parser.add_argument('--sentence_num', type=int, default=4)
parser.add_argument('--paragraph_num', type=int, default=128)
parser.add_argument('--sentence_length', type=int, default=-1)
parser.add_argument('--adversarial', type=int, default=1)

args = parser.parse_args()
