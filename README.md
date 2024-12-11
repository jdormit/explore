# explore
> Interactively explore a codebase with an LLM

[![PyPI - Version](https://img.shields.io/pypi/v/explore-cli?pypiBaseUrl=https%3A%2F%2Fpypi.org)](https://pypi.org/project/explore-cli/)

`explore` is a script to interactively explore a codebase by chatting with an LLM. It uses [retrieval-augmented generation](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) via [`chromadb`](https://docs.trychroma.com/) to provide the LLM with relevant source code from the codebase.

`explore` uses OpenAI models by default, so you'll need an [OpenAI API key](https://openai.com/index/openai-api/).

## Installation
`explore` is available [on PyPI](https://pypi.org/project/explore-cli/). I recommend installing it with [`pipx`](https://github.com/pypa/pipx):

```sh
pipx install explore-cli
export OPENAI_API_KEY=<your OpenAI API key>
explore <directory>
```

Alternatively, you can clone this repository and run the script with [`poetry`](https://python-poetry.org/):

```sh
poetry install
poetry build
export OPENAI_API_KEY=<your OpenAI API key>
poetry run explore <directory>
```

## Usage

```sh
usage: explore [-h] [-l LLM] [-m MODEL] directory

Interactively explore a codebase with an LLM.

positional arguments:
  directory             The directory to index and explore.

options:
  -h, --help            show this help message and exit
  -l LLM, --llm LLM     The LLM backend, one of openai, ollama, or azure. Default: openai. If using Azure, make sure to
                        set the AZURE_OPENAI_ENDPOINT and OPENAI_API_VERSION environment variables.
  -m MODEL, --model MODEL
                        The LLM model to use. Default: gpt-4o-mini for openai, mistral-nemo:latest for ollama, or
                        gpt-4o for azure.
```

## How it works
1. The codebase is indexed into a local Chroma store. Each file is split into chunks using language-specific separators.
2. Documents relevant to the query are collected using multiple retrieval strategies:
   - Primary retrieval is done through vector similarity using the indexed embeddings.
   - A multi-query retriever issues multiple variations of the query to increase the diversity and relevance of retrieved documents.
   - Additionally, a history-aware retriever reformulates the user query, considering the conversation history to better capture context.
3. Retrieved documents are deduplicated, concatenated, and added as context to the LLM, which generates an answer to the user's question. Answers include specific references to the files and code pertinent to the query.

## Using Azure OpenAI
`explore` can connect to an Azure OpenAI instance using Azure Active Directory authentication. First, set the relevant environment variables (you can find the values to use for these in the Azure portal):

``` sh
export AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com # <- endpoint for the Azure OpenAI instance
export OPENAI_API_VERSION=2024-10-01-preview # <- API version for the deployment you want to use
```

When you invoke `explore`, pass the Azure OpenAI deployment name as the `--model` argument and specify `azure` as the `--llm`:

``` sh
$ explore --llm azure --model gpt-4o /some/directory
```
