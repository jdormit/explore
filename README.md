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
  -l LLM, --llm LLM     The LLM backend, one of openai or ollama. Default: openai
  -m MODEL, --model MODEL
                        The LLM model to use. Default: gpt-4o-mini for openai or mistral-nemo:latest for ollama
```
