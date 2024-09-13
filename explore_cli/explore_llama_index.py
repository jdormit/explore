import argparse
import os
from pathlib import Path
import logging
import sys
import chromadb
import chromadb.config
import gnureadline as readline
from httpx import stream
import llama_index.core
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    get_response_synthesizer
)
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine, MultiStepQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)


CONFIG_PATH = os.path.join(str(Path.home()), ".explore")
CHROMADB_PATH = os.path.join(CONFIG_PATH, "db")
HISTORY_FILE = os.path.join(CONFIG_PATH, "history")


def file_metadata(file_path):
    return {
        "filename": file_path,
    }


def main():
    Settings.llm = Ollama(model="mistral-nemo:latest")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # Settings.embed_model = HuggingFaceEmbedding(model_name="dunzhang/stella_en_1.5B_v5")

    os.makedirs(CONFIG_PATH, exist_ok=True)
    os.makedirs(CHROMADB_PATH, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Interactively explore a codebase with an LLM"
    )
    parser.add_argument("directory", help="Path to the codebase to explore.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=3,
        help="Number of context documents to pass to the LLM.",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        llama_index.core.set_global_handler("simple")

    readline.read_history_file(HISTORY_FILE)

    directory = os.path.abspath(os.path.expanduser(args.directory))
    collection_name = directory.replace("/", "_").strip("_")

    # TODO: exclude files in gitignore
    exclude = [".git/**/*", "node_modules/**/*", "tmp/**/*"]
    # TODO: rather than keep every document in memory, iterate through them and add them to the chroma store one by one
    documents = SimpleDirectoryReader(
        directory,
        recursive=True,
        filename_as_id=True,
        exclude=exclude,
        exclude_hidden=False,
        file_metadata=file_metadata,
    ).load_data()
    chromadb_client = chromadb.PersistentClient(
        path=CHROMADB_PATH,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    chroma_collection = chromadb_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store, show_progress=args.verbose)
    print("Indexing codebase...")
    index.refresh(documents)

    # Potential improvement:
    # Perform the vector search as usual and also a keyword search,
    # then rerank the results using https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/#llm-rerank

    # hyde = HyDEQueryTransform(include_original=True)
    query_engine = index.as_query_engine(similarity_top_k=args.top_k)
    # hyde_engine = TransformQueryEngine(query_engine, query_transform=hyde)
    step_decompose_transform = StepDecomposeQueryTransform(verbose=args.verbose)
    response_synthesizer = get_response_synthesizer(streaming=True)
    multi_step_engine = MultiStepQueryEngine(
        query_engine=query_engine,
        query_transform=step_decompose_transform,
        index_summary="Used to answer questions about the codebase",
        response_synthesizer=response_synthesizer
    )
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=multi_step_engine, verbose=args.verbose
    )

    while True:
        question = input("Ask a question about the codebase (or type 'exit' to quit): ")
        if question == "exit":
            break
        print("\n")
        response = chat_engine.stream_chat(question)
        for token in response.response_gen:
            print(token, end="")
        print("\n")
        readline.write_history_file(HISTORY_FILE)


if __name__ == "__main__":
    main()
