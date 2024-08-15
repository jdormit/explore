import os
import logging

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "ERROR").upper())

import argparse
import chromadb
import hashlib
import gnureadline as readline
import re
import textwrap

from collections import Counter
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from fnmatch import fnmatch
from openai import OpenAI
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from tqdm import tqdm

FILE_MAX_LEN = 10000
IGNORED_PATTERNS = [
    "**/.git/*",
    "*.tmp",
    "*.log",
    "*.swp",
    "*.bak",
    "**/node_modules/*",
    "*.sock",
]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

HISTORY_FILE = os.path.join(os.getenv("HOME"), ".explore", "history")

logger = logging.getLogger()

# disable huggingface tokenizers parallelism, it was giving a warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=OPENAI_BASE_URL)

chromadb_n_results = int(os.getenv("CHROMADB_N_RESULTS", 4))
db_path = os.path.join(os.getenv("HOME"), ".explore", "db")
os.makedirs(db_path, exist_ok=True)
client = chromadb.PersistentClient(
    path=db_path, settings=Settings(anonymized_telemetry=False)
)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

messages = []


def extract_keywords(text):
    query = text.strip("?")
    tokens = re.findall(r"\w+", query)
    filtered_tokens = [
        t for t in tokens if t not in ["the", "is", "in", "and", "to", "a", "an", "of"]
    ]
    keywords = {k[0] for k in Counter(filtered_tokens).most_common()}
    keywords = keywords.union({k.lower() for k in keywords})
    return list(keywords)


def load_gitignore(directory):
    gitignore_path = os.path.join(directory, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            return PathSpec.from_lines(GitWildMatchPattern, f)
    return None


def index_directory(directory, ignore_gitignore=True, disable_progress_bar=False):
    collection_name = os.path.basename(os.path.normpath(directory))
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )

    pathspec = load_gitignore(directory) if ignore_gitignore else None

    count_progress = tqdm(desc="Collecting files", unit=" files", disable=disable_progress_bar)
    files = []
    for root, _, dir_files in os.walk(directory):
        for file in dir_files:
            if not (
                any(
                    fnmatch(os.path.join(root, file), pattern)
                    for pattern in IGNORED_PATTERNS
                )
                or (pathspec and pathspec.match_file(file))
            ):
                files.append(os.path.join(root, file))
                count_progress.update(1)

    count_progress.close()

    progress_bar = tqdm(
        total=len(files), desc="Indexing files", unit=" files", miniters=1, disable=disable_progress_bar
    )

    for file_path in files:
        doc_id = hashlib.md5(file_path.encode("utf-8")).hexdigest()
        modified_time = os.path.getmtime(file_path)
        get_res = collection.get(ids=[doc_id], include=["metadatas"], limit=1)
        if (
            len(get_res["ids"]) > 0
            and get_res["metadatas"][0].get("modified_time", -1.0) == modified_time
        ):
            progress_bar.update(1)
            continue
        with open(file_path, "r") as f:
            try:
                content = f"{file_path}:\n\n{f.read()}"
                collection.upsert(
                    documents=[content],
                    ids=[doc_id],
                    metadatas=[{"path": file_path, "modified_time": modified_time}],
                )
                progress_bar.update(1)
            except UnicodeDecodeError:
                logger.warning(f"Invalid UTF-8: {file_path}. Skipping")
                progress_bar.update(1)
    progress_bar.close()
    return collection


def retrieve_documents(collection, question):
    initial_results = collection.query(
        query_texts=[question],
        n_results=chromadb_n_results,
    )
    initial_documents = initial_results["documents"][0]
    fetched_ids = [meta["path"] for meta in initial_results["metadatas"][0]]
    logger.debug(f"Query documents: {fetched_ids}")

    conversation_history = " ".join(msg["content"] for msg in messages)
    additional_results = collection.query(
        query_texts=[conversation_history],
        n_results=3,
        where={"path": {"$nin": fetched_ids}},
    )

    additional_documents = additional_results["documents"][0]
    additional_fetched_ids = [
        meta["path"] for meta in additional_results["metadatas"][0]
    ]
    logger.debug(f"Context documents: {additional_fetched_ids}")

    search_keywords = extract_keywords(question)
    keyword_results = collection.get(
        limit=2,
        where_document={"$or": [{"$contains": keyword} for keyword in search_keywords]},
        where={"path": {"$nin": fetched_ids + additional_fetched_ids}},
    )
    keyword_documents = keyword_results["documents"]
    keyword_fetched_ids = [meta["path"] for meta in keyword_results["metadatas"]]
    logger.debug(f"Keyword documents: {keyword_fetched_ids}")

    return initial_documents + additional_documents + keyword_documents


def query_codebase(collection, question, documents):
    context_documents = "\n\n".join(
        [textwrap.shorten(doc, width=FILE_MAX_LEN) for doc in documents]
    )

    messages.append({"role": "user", "content": question})
    response_text = ""
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert in understanding and explaining code. You will be asked a question about a codebase, respond concisely.\n\nRelevant source files: {context_documents}",
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
    parser = argparse.ArgumentParser(
        description="Interactively explore a codebase with an LLM."
    )
    parser.add_argument("directory", help="The directory to index and explore.")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="skip indexing the directory (warning: if the directory hasn't been indexed at least once, it will be indexed anyway)",
    )
    parser.add_argument(
        "--no-ignore", action="store_true", help="Disable respecting .gitignore files"
    )
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="Only print documents, then exit. --question must be provided",
    )
    parser.add_argument(
        "--question", help="Initial question to ask (will prompt if not provided)"
    )
    parser.add_argument("--no-progress-bar", help="Disable progress bar", action="store_true")
    parser.add_argument("--index-only", help="Only index the directory", action="store_true")
    args = parser.parse_args()

    directory = args.directory
    ignore_gitignore = not args.no_ignore
    documents_only = args.documents_only
    initial_question = args.question
    no_progress_bar = args.no_progress_bar
    index_only = args.index_only

    try:
        if args.skip_index:
            collection_name = os.path.basename(os.path.normpath(directory))
            try:
                collection = client.get_collection(name=collection_name)
            except ValueError:
                print(
                    f"Warning: No existing collection for {directory}. Indexing is required."
                )
                collection = index_directory(directory, ignore_gitignore, disable_progress_bar=no_progress_bar)
        else:
            collection = index_directory(directory, ignore_gitignore, disable_progress_bar=no_progress_bar)

        if index_only:
            return

        if not os.path.exists(HISTORY_FILE):
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, "wb") as f:
                pass  # create the file
        readline.read_history_file(HISTORY_FILE)
        looped = False
        while True:
            if initial_question and not looped:
                question = initial_question
            else:
                question = input(
                    "Ask a question about the codebase (or type 'exit' to quit): "
                )
            looped = True
            if question.lower() == "exit":
                break

            print("", flush=True)
            documents = retrieve_documents(collection, question)
            if documents_only:
                for doc in documents:
                    print(doc)
                break
            for part in query_codebase(collection, question, documents):
                print(part, end="", flush=True)
            print()  # For a new line after the full response
            readline.write_history_file(HISTORY_FILE)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

# TODO:
# - integrate with Emacs
