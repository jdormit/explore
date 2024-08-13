import os
import sys
from openai import OpenAI
import chromadb
import logging
import textwrap

FILE_MAX_LEN = 10000

logger = logging.getLogger()

# Suppress logging
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.ERROR)
# and disable huggingface tokenizers parallelism, it was giving a warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize OpenAI API
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Chroma client with the updated path
db_path = os.path.join(os.getenv("HOME"), ".explore", "db")
os.makedirs(db_path, exist_ok=True)
client = chromadb.PersistentClient(path=db_path)

messages = []


def index_directory(directory):
    collection_name = os.path.basename(os.path.normpath(directory))
    collection = client.get_or_create_collection(name=collection_name)
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                try:
                    print(".", end="", flush=True)
                    content = f.read()
                    document_id = file_path.replace(directory, "")
                    collection.add(
                        documents=[content],
                        ids=[document_id],
                        metadatas=[{"path": file_path}],
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
    context_documents = "\n\n".join(
        [
            f"{doc[0]['path']}:\n{textwrap.shorten(doc[1], width=FILE_MAX_LEN)}"
            for doc in zip(results["metadatas"][0], results["documents"][0])
        ]
    )

    messages.append({"role": "user", "content": question})
    response_text = ""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
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
