# Question-Answer Task Fine-Tuning

This examples shows how to train a Question-Answering Model using HuggingFace.

All the implementation is in the notebook [`qa_finetuning.ipynb`](./qa_finetuning.ipynb).

The example was taken from one of the lectures of the [Udacity Generative AI Nanodegree](), and it uses several snippets from the repository of the [HuggingFace Examples](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/trainer_qa.py).

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

As we can see, the QA example is *extractive* (the answer is in the text), and not *abstractive* (the answer is deduced).

Our *dummy* dataset is about AP controllers (Access Point) -- it is very technical.

The dataset consists of 10 QA pairs in [`data/qa.csv`](./data/qa.csv); in that dataframe, we have three columns

- `question`
- `answer`
- `filename`: this field points to any of two TXT documents ([`CVE-2020-29583.txt`](./data/CVE-2020-29583.txt) and [`xss.txt`](./data/xss.txt)), where the context for the answer is provided.

We transform the dataset CSV to the [SQuAD 2.0](https://arxiv.org/abs/1806.03822) format above, i.e., a list of dictionaries.

## Model


