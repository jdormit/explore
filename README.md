# explore
> Interactively explore a codebase with an LLM

`explore` is a script to interactively explore a codebase by chatting with an LLM. It uses [retrieval-augmented generation](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) via [`chromadb`](https://docs.trychroma.com/) to provide the LLM with relevant source code from the codebase.

`explore` uses OpenAI models by default, so you'll need an [OpenAI API key](https://openai.com/index/openai-api/).

## Installation
`explore` is available [on PyPI](). I recommend installing it with [`pipx`](https://github.com/pypa/pipx):

``` sh
pipx install explore-cli
explore <directory>
```

Alternatively, you can clone this repository and run the script with [`poetry`](https://python-poetry.org/):

``` sh
poetry install
poetry build
poetry run explore <directory>
```

## Usage

``` sh
usage: explore [-h] [--skip-index] [--no-ignore] [--documents-only] [--question QUESTION] [--no-progress-bar]
               [--index-only]
               directory

Interactively explore a codebase with an LLM.

positional arguments:
  directory            The directory to index and explore.

options:
  -h, --help           show this help message and exit
  --skip-index         skip indexing the directory (warning: if the directory hasn't been indexed at least once, it
                       will be indexed anyway)
  --no-ignore          Disable respecting .gitignore files
  --documents-only     Only print documents, then exit. --question must be provided
  --question QUESTION  Initial question to ask (will prompt if not provided)
  --no-progress-bar    Disable progress bar
  --index-only         Only index the directory
```

## Configuration
There are a couple of environment variables you can set to configure `explore`:
| Name  | Description  |
|---|---|
| `OPENAI_BASE_URL`  | The base URL used for OpenAI API requests. You can set this to use any OpenAI-compatible APIs (e.g. [Ollama](https://ollama.com/blog/openai-compatibility) to run models locally). Default: `https://api.openai.com/v1` |
| `OPENAI_MODEL`  |  Which model to tell the OpenAI API to use. The default is `gpt-4o-mini`, which strikes a good balance between coherence and price. You can get better results if you set this to `gpt-4o`, but bear in mind `explore` can generate extremely long prompts so that could get expensive quickly |


## How it works
1. The directory is indexed into a local Chroma store. Only files that have been modified since the last time they were indexed get re-indexed, so this step will be quite slow on the first execution but pretty quick after that.
2. Documents relevant to the query are collected in three ways:
   1. The question is embedded as a vector and used to search for the nearest matches in the Chroma DB
   2. The entire conversation so far is embedded as a vectore and used to search for more matches in Chroma
   3. Search keywords are extracted from the question and used to find exact matching text in the indexed documents
   
   By default, `explore` will fetch 4 documents using the first approach, 3 using the second and 2 using the third.
3. The documents are all concatenated and prepended to the ongoing conversation, then the latest question is appended. The whole thing is sent to the LLM, which returns an answer to the question based on the provided documents.
