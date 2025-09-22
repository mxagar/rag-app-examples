# Question-Answer Task Fine-Tuning

This examples shows how to train a Question-Answering Model using HuggingFace.

All the implementation is in the notebook [`qa_finetuning.ipynb`](./qa_finetuning.ipynb).

The example was taken from one of the lectures of the [Udacity Generative AI Nanodegree](https://www.udacity.com/course/generative-ai--nd608), and it uses several snippets from the repository of the [HuggingFace Examples](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/trainer_qa.py).

## Setup

Create a python environment and install the dependencies:

```bash
# Create the necessary Python environment
conda env create -f conda.yaml
conda activate gennai

# Compile and install all dependencies
pip-compile requirements.in
pip-sync requirements.txt

# If we need a new dependency,
# add it to requirements.in 
# And then:
pip-compile requirements.in
pip-sync requirements.txt
```

## Dataset

The example uses the [SQuAD 2.0](https://arxiv.org/abs/1806.03822) format, in which each QA pair is formatted as follows

```python
{
  'id': 'xxx',
  'title': 'my title',
  'context': 'Here the complete context or text document is added.', # our document
  'question': 'What is...?', # our question
  'answers': {
    'text': ['1925'], # list of answers (list(str))
    'answer_start': [354] # the chars in context where the answer text starts (list(int))
  }
}
```

As we can see, the QA example is *extractive* (the answer is in the text), and not *generative-abstractive* (the answer is deduced).

Our *dummy* dataset is about AP controllers (Access Point) -- it is very technical.

The dataset consists of 10 QA pairs in [`data/qa.csv`](./data/qa.csv); in that dataframe, we have three columns

- `question`
- `answer`
- `filename`: this field points to any of two TXT documents ([`CVE-2020-29583.txt`](./data/CVE-2020-29583.txt) and [`xss.txt`](./data/xss.txt)), where the context for the answer is provided.

We transform the dataset CSV to the [SQuAD 2.0](https://arxiv.org/abs/1806.03822) format above, i.e., a list of dictionaries.

## Model

Since we are working on a *extractive* QA task, we can use any encoder or decoder transformer as the backbone; we add a small QA head on top, a linear layer that predicts:

- `start_logits`: probabilities of each token being the start of the answer
- and `end_logits`: probabilities of each token being the end of the answer

The selected model is [distilbert](https://huggingface.co/docs/transformers/en/model_doc/distilbert), and we build a custom `QuestionAnsweringTrainer(Trainer)` for training it, based on the official HuggingFace example [here](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/trainer_qa.py).

## Pre-Processing and Post-Processing Functions

Pre- and post-processing functions are required to adapt the data and the outputs; they are taken from the HuggingFace `run_qa.py` example [on Github](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py):

1. `prepare_train_features(...)`: This function prepares raw QA examples for training by:
   - Tokenizing each (question, context) pair with padding and truncation.
   - Using return_offsets_mapping to trace back token positions to character positions in the context.
   - For each example:
     - Computes the start and end token indices of the answer span using the offsets.
     - If there's no answer, sets both positions to the CLS token index (for null prediction).
   - Returns tokenized inputs with additional "start_positions" and "end_positions" fields used for model training.
2. `postprocess_qa_predictions(...)`: This function transforms the model output logits into human-readable answer strings by:
   - Mapping predicted start and end logits back to their char spans in the original context using offset mappings.
   - Collecting top n-best candidate spans per feature based on score (start_logit + end_logit).
   - Filtering out invalid spans (too long, reversed, or not in max context).
   - Optionally handles "null answers" (no answer present) when version_2_with_negative=True.
   - Returns:
     - all_predictions: best answer per example.
     - all_nbest_json: top-n candidates per example.
     - Optionally, scores_diff_json: for null answer thresholding.
3. `post_processing_function(examples, features, predictions, stage="eval")`: A wrapper around `postprocess_qa_predictions`, specifically used in evaluation or inference, that:
   - Calls `postprocess_qa_predictions()` with sensible defaults.
   - Converts the final predictions to the format expected by HuggingFace metrics (e.g. "id", "prediction_text").
   - Prepares the reference answers in expected format (for metric computation like exact match or F1).
   - Returns an `EvalPrediction` object from the Transformers library.

## Training

After mapping the `prepare_train_features` pre-processing function to the dataset, we train the model with `QuestionAnsweringTrainer`.

## Inference

To run inference, we can use the `pipeline` function from HuggingFace, which in our case uses/returns a `QuestionAnsweringPipeline` object.

Note that DistilBert has a context window of 512 tokens only, whereas our contexts surpass that size; `QuestionAnsweringPipeline` handles that automatically by chunking the context with some overlap and running the model with each context. Then, the best answer (the one with the highest score) is chosen.

We can also write a custom `ask()` function that does that.

