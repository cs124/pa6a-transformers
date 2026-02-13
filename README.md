#  Programming Assignment 6a - Transformers

In this assignment, you will implement part of a Transformer-based language model for text generation and train it on a dataset of Shakespeare corpus. 
You will then use your trained model to generate some texts.

## Note on PyTorch
In this assignment, we will use [PyTorch](https://docs.pytorch.org/docs/stable/index.html), 
an open-source library widely used for training neural networks and large language models. 
Like NumPy, it supports efficient matrix computations, and it also provides automatic 
differentiation for computing gradients. If you'd like to familiarize yourself with PyTorch 
before diving in, we recommend the CS224N PyTorch tutorial [here](https://colab.research.google.com/drive/1CAO17E5ikaAPw3O7dMKp5TkHYHJE3EJL?usp=sharing). 

We have provided scaffolding and hints throughout the assignment to guide you through each step.

## Cloning the Assignment

Open up the terminal (open PowerShell and type `wsl` if you are on Windows, or `ssh` into Myth/Rice if you are using a remote machine) and execute the following commands:

```
cd folder/to/put/assignment
git clone https://github.com/cs124/pa6a-transformers.git
```

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

After training your model and generating sampled sentences, you will implement the `compute_perplexity` function in `perplexity.py`. This function takes model logits and target tokens as input and returns the perplexity of the sequence. Look for the `# TODO` comment and follow the step-by-step instructions in the docstring.

You can verify your implementation by running the provided unit tests:
```
python test_perplexity.py
```

These tests use synthetic inputs (no trained model needed) to check that your math is correct. Make sure all tests pass before proceeding.

Once your implementation passes the tests, run the full perplexity script to compute and save results for the test sentences:
```
python perplexity.py
```

This will save the results into a json file called `perplexity_results.json`.

**Important:** Do not change the provided list of test sentences or the saving format, as our grading script depends on them.

## Part 4: Ethics

In Part 3, you calculated the perplexity of your model. Many commercial AI detectors use perplexity (i.e., the predictability of a word sequence) and burstiness (i.e, the variation in sentence structure and perplexity throughout a text) to determine if a document was written by a human or an AI. Notably, the sampling methods used during LLM text generation tend to flatten probability distributions, making outputs more uniform and predictable.

Read this article on [Why Perplexity and Burstiness Fail to Detect AI](https://www.pangram.com/blog/why-perplexity-and-burstiness-fail-to-detect-ai) and consider your own model's output.

Open `ethics_responses.txt` and write your responses directly below each question. The questions are:

1. **Defining Human Writing:** Perplexity measures how surprising a sequence of words is. By using this as a detection metric, we are effectively defining human creativity as unpredictability.
   - a) Do you think unpredictability captures something that resonates with your personal understanding of what makes writing human? Why or why not?
   - b) Either way, who might be unfairly impacted by a tool that flags predictable prose as non-human?

2. **The Reward for Standardization:** Reflect on your own education.
   - a) Have you ever been encouraged or rewarded for adopting a highly structured, standard, or predictable writing style (e.g., for a standardized test or a specific rubric)?
   - b) What tensions emerge when the writing style that earns academic success is also the style that AI detectors flag as suspicious?
   - c) Should educators use these AI detector tools?
   - d) What alternatives should educators consider if they are concerned about AI writing in classes meant to teach writing skills?

3. **The Incentive Loop:** As we increasingly use AI-based tools (like Grammarly or LLM editors) to "polish" our work, our writing naturally becomes more predictable.
   - a) What incentives does this create for different stakeholders (e.g., students, educators, institutions, etc.)?
   - b) If detection becomes mathematically harder, what alternative frameworks might we need to think about academic integrity and the role of writing in learning?

## Part 5: Zip and Submit 

Run `bash create_assignment_zip.sh` to zip your submission. The script will create a zip file in your PA6a folder. Submit the zip file to Gradescope.

To recap, the submission zip should include the following files:

- `model.py`: your implementation of the Transformer model
- `perplexity.py`: your implementation of the perplexity calculation
- `perplexity_results.json`: the perplexity results of your sampled sentences
- `sampled_sentences.json`: your sampled sentences
- `ethics_responses.txt`: your responses to the ethics questions
