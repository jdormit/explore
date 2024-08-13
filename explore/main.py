import os
import sys
from openai import OpenAI
import chromadb

# Initialize OpenAI API
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Chroma client with the updated path
db_path = os.path.join(os.getenv('HOME'), '.explore', 'db')
os.makedirs(db_path, exist_ok=True)
client = chromadb.PersistentClient(path=db_path)

def index_directory(directory):
    collection_name = os.path.basename(os.path.normpath(directory))
    collection = client.get_or_create_collection(name=collection_name)
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                try:
                    content = f.read()
                    document_id = file_path.replace(directory, "")
                    collection.add(
                        documents=[content],
                        ids=[document_id],
                        metadatas=[{"path": file_path}]
                    )
                except UnicodeDecodeError:
                    print(f"Invalid UTF-8: {file_path}. Skipping")
    return collection

def query_codebase(collection, question):
    # Step 1: Retrieve relevant documents
    results = collection.query(
        query_texts=[question],
        n_results=5,
    )
    context_documents = results['documents'][0]

    # Step 2: Combine context with the question for the LLM
    combined_input = "\n\n".join(context_documents) + "\n\nQuestion: " + question

    # Step 3: Send the combined input to the OpenAI Chat API with streaming
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in understanding and explaining code."},
            {"role": "user", "content": combined_input}
        ],
        stream=True
    )

    for chunk in response:
        if len(chunk.choices) > 0:
            yield (chunk.choices[0].delta.content or "")

def main():
    if len(sys.argv) < 2:
        print("Usage: explorer <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    collection = index_directory(directory)
    print("Directory indexed.")

    while True:
        question = input("Ask a question about the codebase (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        print("Answer:", end=" ", flush=True)
        for part in query_codebase(collection, question):
            print(part, end="", flush=True)
        print()  # For a new line after the full response

if __name__ == "__main__":
    main()

# TODO:
# - only re-index changed files (by calculating a hash of the file contents and saving it as metadata)
# - save conversational history back into the prompt
# - integrate with Emacs
# - disable warnings/logging from chromadb
