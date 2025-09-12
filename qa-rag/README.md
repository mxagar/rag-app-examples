# Simple Retrieval Augmented Generation (RAG) from Scratch


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

[`gpt-3.5-turbo-instruct`](https://platform.openai.com/docs/models/gpt-3.5-turbo?snapshot=gpt-3.5-turbo-instruct)

- Legacy GPT model for cheaper chat and non-chat tasks
- 4,096 context window
- Sep 01, 2021 knowledge cutoff

Dataset: [2024 Events in Spain (Wikipedia)](https://en.wikipedia.org/wiki/2024_in_Spain)

| Column | Description |
| --- | --- |
| `date` | Date of the event (parsed from the Wikipedia bullet or section) |
| `event` | Short description (from the bullet in the Wikipedia article). |
| `reference_content` | The actual **fetched and cleaned content** from the reference (e.g. paragraph from news article or Wikipedia page). |
| `reference_url` | The URL used to fetch `reference_content`. Can be a news source, Wikipedia article, etc. |
| `reference_entity` | Human-readable source label: e.g. `Wikipedia`, `Reuters`, `AP`, `El País`, etc. Extracted from domain or metadata. |
| `text` | `event` + `reference_content`, used for search and context creation. |

