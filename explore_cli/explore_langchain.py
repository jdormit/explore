import argparse
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings


def collection_name(directory):
    return directory.replace("/", "_").strip("_")


def collect_documents(directory):
    loader = DirectoryLoader(
        directory,
        glob="**/*",
        exclude=[
            "**/.git/*",
            "*.tmp",
            "*.log",
            "*.swp",
            "*.bak",
            "**/node_modules/*",
            "**/.venv/*",
            "*.sock",
        ],
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        silent_errors=True,
        use_multithreading=True,
        show_progress=True,
    )
    # TODO: skip documents that haven't changed
    # TODO: split documents into chunks
    return loader.load()


def main():
    parser = argparse.ArgumentParser(
        description="Interactively explore a codebase with an LLM."
    )
    parser.add_argument("directory", help="The directory to index and explore.")

    args = parser.parse_args()
    directory = os.path.abspath(os.path.expanduser(args.directory))

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2", show_progress=True
    )
    vector_store = Chroma(
        collection_name=collection_name(directory),
        embedding_function=embedding_model,
        persist_directory=os.path.join(os.getenv("HOME"), ".explore", "db-langchain"),
    )
    docs = collect_documents(directory)
    for doc in docs:
        print(doc.metadata["source"])
    # create chroma store from persisted path
    # index any changed/new documents
    # create chat engine
    # run chat in loop


if __name__ == "__main__":
    main()
