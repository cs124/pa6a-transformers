"""
Compute perplexity of input sentences using a trained model checkpoint.
"""
import os
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import numpy as np
import json
# -----------------------------------------------------------------------------
init_from = 'resume'
input_sentences = [
    "To be, or not to be: that is the question.",
    "O Romeo, Romeo! wherefore art thou Romeo?",
    "The lady doth protest too much, methinks.",
    "purple toaster gallops software tranquility penguin umbrella.",
    "kzvtr qwam snejk gurt ploov zex."
]
device = 'cpu' # or 'cuda' as needed
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# -----------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def load_model_and_tokenizer(out_dir):
    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)

    # Tokenizer setup
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return model, encode

def compute_perplexity(logits, y_true):
    """
    Compute perplexity given model logits and true target tokens.

    Args:
        logits: Tensor of shape [1, T, vocab_size] — raw (unnormalized) model outputs.
        y_true: Tensor of shape [1, T] — the true next-token ids.

    Returns:
        ppl: A scalar float — the perplexity of the sequence.
    """
    ppl = None
    # TODO: Compute the perplexity of the input sentence given the model's outputs.
    # What is perplexity?
    #   - Perplexity is a measure of how well a language model predicts a sequence.
    #   - Lower perplexity indicates the model is more confident about its predictions.
    #   - For a sequence of tokens x1, x2, ..., xn, the per-token perplexity is:
    #         exp(mean negative log-likelihood of the target tokens under the model)
    #
    # Steps:
    # 1. Compute the log-probabilities (log softmax) of the logits along the vocabulary dimension.
    # 2. For each position, select the log-probability of the *true* next token from the model output.
    #    - For the t-th position, the target is y_true[0, t].
    #    - Use torch.arange to index positions.
    # 3. Compute the average negative log-probability (i.e., mean negative log-likelihood).
    # 4. Take the exponential to get perplexity.
    #
    # Hints:
    #   - `logits` has shape [1, T, vocab_size].
    #   - `y_true` has shape [1, T].
    #   - Use torch.log_softmax.
    #   - Index with [0, torch.arange(T), y_true[0]].

    # YOUR CODE HERE
    pass

    return ppl

def perplexity_sentence(model, encode, sentence):
    # Encode the sentence and get tokens
    tokens = encode(sentence)
    if len(tokens) < 2:
        return float('nan')  # Not enough tokens to compute perplexity
    x = torch.tensor(tokens[:-1], dtype=torch.long, device=device)[None, ...] # [1, T-1]
    y_true = torch.tensor(tokens[1:], dtype=torch.long, device=device)[None, ...] # [1, T-1]
    with torch.no_grad():
        with ctx:
            logits, _ = model(x, y_true)
            ppl = compute_perplexity(logits, y_true)
    return ppl

if __name__ == "__main__":
    out_dir = 'out-shakespeare'
    output_file = 'perplexity_results.json'
    results = []
    print(f"Loading model from {out_dir}...")
    model, encode = load_model_and_tokenizer(out_dir)
    print(f"Perplexity for {out_dir}:")
    for sent in input_sentences:
        ppl = perplexity_sentence(model, encode, sent)
        results.append({
            'sentence': sent,
            'perplexity': ppl
        })
        print(f"Perplexity for: \"{sent}\"")
        print(f"Perplexity: {ppl:.3f}")
        print("-" * 50)

    # save the results to a json file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)