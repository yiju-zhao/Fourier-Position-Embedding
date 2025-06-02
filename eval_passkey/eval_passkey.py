import argparse
import random
import re
import sys
import torch
import warnings
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate

# from https://github.com/epfml/landmark-attention/blob/main/llama/run_test.py

sys.path.append("Fourier-Position-Embedding")
from olmo.model import OLMoForCausalLM # type: ignore
from olmo.tokenizer import Tokenizer # type: ignore

import os


def generate_prompt(n_garbage):
    """Generates a text file and inserts an execute line at a random position."""
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 10000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key


def test_model(model, tokenizer, prompt_text, pass_key):
    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    
    try:
        outputs = model.generate(input_ids, max_new_tokens=10, eliminate_token_ids=False)
        output_str = tokenizer.decode(outputs[0])
        response = output_str[len(prompt_text):]
        
    except:
        response = "null"
    
    assert f"The pass key is {pass_key}" in prompt_text

    try:
        pass_key = int(re.search(r'\d+', response).group())
    except:
        pass_key = response[:20]

    return pass_key

def load_model(model_path, args):
    device = torch.device(f"cuda:{args.device}" if args.cuda else "cpu")
    model = OLMoForCausalLM.from_checkpoint(model_path, device=device)
    if args.cuda:
        model.to(torch.device(f"cuda:{args.device}"))
    
    return model


def main(args):
        
    models = ["model_path1", "model_path2"]  # Replace with actual model paths
    
    tokenizer_path = "path_to_olmo_data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"
    tokenizer = Tokenizer.from_file(
        tokenizer_path,
        eos_token_id=50256, 
        pad_token_id=50256,
        pad_direction="right" # pad_direction="left"
    )

    if args.fixed_length:
        lengths = [args.fixed_length]
        tokens = [len(tokenizer.encode(generate_prompt(args.fixed_length)[0]))]
        print(f"Prompt is {tokens[0]} tokens")
    else:
        if args.tokens_step:
            tokens = [x for x in range(
                args.min_tokens, args.max_tokens + 1, args.tokens_step)]
        else:
            tokens = [args.min_tokens]
            while args.min_tokens < args.max_tokens:
                point = tokens[-1] * 2
                if point <= args.max_tokens:
                    tokens.append(point)
                else:
                    break

        lengths = []
        last_n = 0
        for target in tqdm(tokens, desc="Determining sequence lengths"):
            num_tokens = 0
            n = last_n
            while num_tokens < target:
                last_n = n
                n += args.length_step
                prompt = generate_prompt(n)[0]
                num_tokens = len(tokenizer.encode(prompt))
            lengths.append(last_n)
    
    if args.same_examples:
        prompt_texts, pass_keys = {}, {}
        for length in lengths:
            prompt_text, pass_key = zip(*[generate_prompt(length) for _ in range(args.iterations)])
            prompt_texts[length], pass_keys[length] = prompt_text, pass_key
    
    results = []
    for model in tqdm(models, desc="Model", leave=False):
        torch.cuda.empty_cache()

        loaded = load_model(model, args)

        result = [0] * len(lengths)
        for i, length in tenumerate(lengths, desc="Testing", leave=False):
            for it in trange(0, args.iterations, desc=f"Length={tokens[i]}", leave=False):
                if not args.same_examples:
                    prompt_text, pass_key = generate_prompt(length)
                else:
                    prompt_text, pass_key = prompt_texts[length][it], pass_keys[length][it]
                
                num_tokens = len(tokenizer.encode(prompt_text))
                
                answer = test_model(loaded, tokenizer, prompt_text, pass_key)
                
                
                if answer == "null":
                    print(f"Error: null response")
                else:
                    if answer == pass_key:
                        result[i] += 1
                    
            result[i] /= args.iterations
            
            print(f"{model}: {tokens[i]} = {result[i]*100:.4f}%")

        result.insert(0, model)
        results.append(result)

    if args.output_file:
        if not os.path.exists(args.output_file):
            with open(args.output_file, 'w', encoding="utf-8") as f:
                pass
            
        with open(args.output_file, "a", encoding="utf-8") as f:
            f.write(f"model,{','.join([str(x) for x in tokens])}\n")
            for result in results:
                f.write(f"{','.join([str(x) for x in result])}\n")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", action="append", nargs="+")
    parser.add_argument("--fixed-length", type=int)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument("--tokens-step", type=int)
    parser.add_argument("--length-step", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--same-examples", action="store_true")
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    
    args = parser.parse_args()
    
    main(args)
