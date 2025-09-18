# Simple Retrieval Augmented Generation (RAG) from Scratch

This small project shows how to build a basic Question-Answering chatbot based on the [Retrieval-Augmented Generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) pattern from scratch.

RAG consists in 

- retrieving relevant documents related to the user **query** 
- and feeding them to a generative model in the context along with the query; 
- finally, we instruct the model to provide the answer using the provided documents/context.

Therefore, we avoid needing to fine-tune the generative model with our documents.
This is specially well suited when we want to extend the model's *memory* with recent and continuously changing documents.

In order to show how the approach works,

- I use the **model** [`gpt-3.5-turbo-instruct`](https://platform.openai.com/docs/models/gpt-3.5-turbo?snapshot=gpt-3.5-turbo-instruct) from OpenAI
- and a **dataset** or set **queried documents** built from the Wikipedia article [2024 Events in Spain](https://en.wikipedia.org/wiki/2024_in_Spain) (54 events in total).

The [`gpt-3.5-turbo-instruct`](https://platform.openai.com/docs/models/gpt-3.5-turbo?snapshot=gpt-3.5-turbo-instruct) model

- is a Legacy GPT model for cheaper chat and non-chat tasks,
- has a context window of `4,096` tokens,
- and has a **knowledge cutoff as of Sep 01, 2021**.

Therefore, we can be sure that none of the 2024 events in Spain were used for during the model training. Therefore,

- if we ask the model a question about the dataset, it should hallucinate and/or fail to answer properly;
- but if we use the RAG pattern, it should be able to build and use a relevant context that facilitates a correct answer.

## Setup

Create an OpenAI account and get an API key; we should save the key in our local and uncommitted `.env`.

Then, create a python environment and install the dependencies:

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

Finally, open the notebook where everything is implemented: [`qa_rag.ipynb`](./qa_rag.ipynb).

## Dataset Indexing

The dataset or queried documents are scrapped from the Wikipedia article [2024 Events in Spain](https://en.wikipedia.org/wiki/2024_in_Spain) using `BeautifulSoup`.

The main parsing function is `get_wikipedia_events()`, which returns a list of dictionaries; each dictionary is an event, which contains: 

- `month (str)`: The month of the event.
- `date_text (str)`: The raw date text from the page.
- `date (datetime.date)`: The parsed start date of the event.
- `date_end (Optional[datetime.date])`: The parsed end date if it's a date range, else `None`.
- `event (str)`: The cleaned event description.
- `refs (list[str])`: List of reference IDs from the page.
- `reference_urls (list[str])`: List of URLs for references.
- `reference_entities (list[str])`: List of entities associated with the references.

Then, the the content in the reference URLs is fetched in `get_event_reference_contents()`, which updated the dictionaries with the field `reference_content`.

Additionally, the field `text` is also added in the same function; `text` which is the concatenation of the `date + event + reference_content` fields, i.e., a detailed description of the event.

Finally, `docs_to_embeddings_df()` converts the event dictionaries into a dataframe with 54 rows (events) and the columns `date`, `event`, `text` and `embedding`.

The `embedding` column contains the semantic embedding the the `text` column, which can be generated using:

- HuggingFace (model `intfloat/e5-large-v2`): `compute_embeddings_hf()`
- or OpenAI (model: `text-embedding-ada-002`): `compute_embeddings_openai()`

The resulting dataframe is the *knowledge base* used for the RAG pipeline.

## Query Pipeline



## Examples

