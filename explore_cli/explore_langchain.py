import argparse
from fnmatch import fnmatch
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPattern

IGNORED_PATTERNS = [
    "**/.git/*",
    "*.tmp",
    "*.log",
    "*.swp",
    "*.bak",
    "**/node_modules/*",
    "*.sock",
]


def collection_name(directory):
    return directory.replace("/", "_").strip("_")


def load_gitignore(directory):
    gitignore_path = os.path.join(directory, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            return PathSpec.from_lines(GitWildMatchPattern, f)
    return None


def collect_documents(directory, use_gitignore=True):
    pathspec = load_gitignore(directory) if use_gitignore else None
    docs = []
    # TODO: skip files that haven't changed
    for root, _, dir_files in os.walk(directory):
        for file in dir_files:
            if not (
                any(
                    fnmatch(os.path.join(root, file), pattern)
                    for pattern in IGNORED_PATTERNS
                )
                or (pathspec and pathspec.match_file(file))
            ):
                try:
                    docs.extend(
                        TextLoader(
                            file_path=os.path.join(root, file), autodetect_encoding=True
                        ).load()
                    )
                except Exception as e:
                    print(f"Error loading {os.path.join(root, file)}: {e}")
    # TODO: split documents into chunks
    return docs


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
