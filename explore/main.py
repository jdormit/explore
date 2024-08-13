import os
import logging

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "ERROR").upper())

import chromadb
import hashlib
import sys
import textwrap

from fnmatch import fnmatch
from openai import OpenAI

FILE_MAX_LEN = 10000
IGNORED_PATTERNS = ["**/.git/*", "*.tmp", "*.log", "*.swp", "*.bak"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

logger = logging.getLogger()

# disable huggingface tokenizers parallelism, it was giving a warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

db_path = os.path.join(os.getenv("HOME"), ".explore", "db")
os.makedirs(db_path, exist_ok=True)
client = chromadb.PersistentClient(path=db_path)

messages = []


def hash_file(file_path):
    """Compute the hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def index_directory(directory):
    collection_name = os.path.basename(os.path.normpath(directory))
    collection = client.get_or_create_collection(name=collection_name)
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if any(fnmatch(file_path, pattern) for pattern in IGNORED_PATTERNS):
                continue
            file_hash = hash_file(file_path)
            get_res = collection.get(where={"hash": file_hash}, include=[], limit=1)
            if len(get_res["ids"]) > 0:
                continue
            with open(file_path, "r") as f:
                try:
                    print(".", end="", flush=True)
                    content = f"{file_path}:\n\n{f.read()}"
                    document_id = file_path.replace(directory, "")
                    collection.upsert(
                        documents=[content],
                        ids=[document_id],
                        metadatas=[{"path": file_path, "hash": file_hash}],
                    )
                except UnicodeDecodeError:
                    logger.warning(f"Invalid UTF-8: {file_path}. Skipping")
    return collection


def query_codebase(collection, question):
    # Step 1: Retrieve relevant documents
    results = collection.query(
        query_texts=[question],
        n_results=5,
    )
    logger.debug(
        f"Using documents: {[meta['path'] for meta in results['metadatas'][0]]}"
    )

    context_documents = "\n\n".join(
        [textwrap.shorten(doc, width=FILE_MAX_LEN) for doc in results["documents"][0]]
    )

    messages.append({"role": "user", "content": question})
    response_text = ""
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert in understanding and explaining code. You will be asked a question about this codebase, respond concisely.\n\nRelevant source files: {context_documents}",
            }
        ]
        + messages,
        stream=True,
    )

    for chunk in response:
        if len(chunk.choices) > 0:
            text = chunk.choices[0].delta.content or ""
            response_text += text
            yield text
    messages.append({"role": "assistant", "content": response_text})


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    print("Indexing directory...")
    collection = index_directory(directory)
    print("")
    print("Directory indexed.")

    while True:
        question = input("Ask a question about the codebase (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        print("\n", flush=True)
        for part in query_codebase(collection, question):
            print(part, end="", flush=True)
        print()  # For a new line after the full response


if __name__ == "__main__":
    main()

# TODO:
# - only re-index changed files (by calculating a hash of the file contents and saving it as metadata)
# - integrate with Emacs
# - disable warnings/logging from chromadb
