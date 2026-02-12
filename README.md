#  Programming Assignment 6a - Transformers

In this assignment, you will implement part of a Transformer-based language model for text generation and train it on a dataset of Shakespeare corpus. 
You will then use your trained model to generate some texts.

## Environment Setup
PA6a requires more packages than the `cs124` base environment you previously created. There are two ways to setup your environment:

1. Activate the `cs124` environment you previously created, then install additionally required packages:

```
conda activate cs124
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

2. Create a new environment called `cs124_pa6a`:

```
conda env create -f environment_pa6a.yml
conda activate cs124_pa6a
```

## Part 1: Implement Attention in Transformer

Check out `model.py` which implements a vanilla Transformer model. 

You need to implement the `forward` method of the `CausalSelfAttention` class based on what you learned in class. You do not need to change anything else in `model.py`, but you are encouraged to read through the implementation to understand the model architecture.

We have provided a simple unit test for the `CausalSelfAttention` class. You can test whether your implementation is correct by running:
```
python test_attention.py
```

## Part 2: Training some nanoGPT models 

Once that's done, you can train the model by running:

```
python train.py
```

We have already processed the Shakespeare corpus and stored the tokenized data in `data/`.
The training script uses the data tokenized via the GPT-2 BPE tokenizer.
The training script runs on CPU by default. Running the training script for 2000 iterations on my Macbook Pro with M1 chip takes around 15 minutes and gets a loss below 4.0.
If you have extra time, you are encouraged to try out different hyperparameters and try to improve the validation loss of your model.

The trained model checkpoints will be automatically saved in the `out-shakespeare/` directory.
You can then load the model checkpoint and sample from the model by running:

```
python sample.py
```

You are encouraged to tweak the sampling hyperparameters to understand the effects of different sampling hyperparameters like the temperature. Our model is trained on a tiny corpus, so the sampled sentences may not be very fluent, but you should be able to see some Shakespearean-like writing patterns and some basic grammar. If you only get gibberish, you probably did something wrong either in the training or sampling process.

Use your trained model to sample 5 sentences with maximum 100 tokens each. As part of the submission, save your sampled sentences as a list of strings in a json file called `sampled_sentences.json`. We will judge the quality of your sampled sentences based on whether they are at least somewhat coherent and grammatically correct.

## Part 3: Implement Perplexity Calculation

After training your model and generating sampled sentences, you will implement code in `perplexity.py` to compute the perplexity of several test sentences using your trained model.

In `perplexity.py`, fill in the section that computes perplexity for an input sentence (look for the `# TODO` comment in the `perplexity_sentence` function). Once your implementation is complete, run the script to compute and save perplexity results into a json file called `perplexity_results.json`.

**Important:** Do not change the provided list of test sentences or the saving format, as our grading script depends on them.

## Part 4: Zip and Submit 

Run `bash create_assignment_zip.sh` to zip your submission. The script will create a zip file in your PA6a folder. Submit the zip file to Gradescope.

To recap, the submission zip should include the following files:

- `model.py`: your implementation of the Transformer model
- `perplexity_results.json`: the perplexity results of your sampled sentences
- `sampled_sentences.json`: your sampled sentences
