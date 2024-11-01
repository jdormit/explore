import argparse
from fnmatch import fnmatch
import gnureadline as readline
import os
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import Language
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPattern
from pydantic_core.core_schema import FieldPlainInfoSerializerFunction
from rich.console import Console
from rich.markdown import Markdown
from sklearn.base import defaultdict

# disable huggingface tokenizers parallelism, it was giving a warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

IGNORED_PATTERNS = [
    "**/.git/*",
    "*.tmp",
    "*.log",
    "*.swp",
    "*.bak",
    "**/node_modules/*",
    "*.sock",
]

LANGUAGES_BY_EXTENSION = {
    "cpp": Language.CPP,
    "go": Language.GO,
    "java": Language.JAVA,
    "kt": Language.KOTLIN,
    "js": Language.JS,
    "ts": Language.TS,
    "php": Language.PHP,
    "proto": Language.PROTO,
    "py": Language.PYTHON,
    "rst": Language.RST,
    "rb": Language.RUBY,
    "rs": Language.RUST,
    "scala": Language.SCALA,
    "swift": Language.SWIFT,
    "md": Language.MARKDOWN,
    "tex": Language.LATEX,
    "html": Language.HTML,
    "sol": Language.SOL,
    "cs": Language.CSHARP,
    "cbl": Language.COBOL,
    "c": Language.C,
    "lua": Language.LUA,
    "pl": Language.PERL,
    "hs": Language.HASKELL,
}


def collection_name(directory):
    return directory.replace("/", "_").strip("_")


def load_gitignore(directory):
    gitignore_path = os.path.join(directory, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            return PathSpec.from_lines(GitWildMatchPattern, f)
    return None


def separators_for_extension(extension):
    if extension in LANGUAGES_BY_EXTENSION:
        return RecursiveCharacterTextSplitter.get_separators_for_language(
            LANGUAGES_BY_EXTENSION[extension]
        )
    if extension == "el":
        return [
            "(use-package",
            "(defun ",
            "(defvar ",
            "(let",
            "(if",
            "\n\n",
            "\n",
            " ",
        ]
    return None


def split_docs(documents):
    split_docs = []
    docs_by_extension = defaultdict(list)
    for doc in documents:
        docs_by_extension[doc.metadata["source"].split(".")[-1]].append(doc)
    for ext, docs in docs_by_extension.items():
        separators = separators_for_extension(ext)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=250,
            length_function=len,
            is_separator_regex=False,
            separators=separators,
        )
        split_docs.extend(splitter.transform_documents(docs))
    return split_docs


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
    return split_docs(docs)


def format_docs(docs):
    return "\n\n".join(
        [f"{doc.metadata['source']}:\n{doc.page_content}" for doc in docs]
    )


def main():
    console = Console()

    parser = argparse.ArgumentParser(
        description="Interactively explore a codebase with an LLM."
    )
    parser.add_argument("directory", help="The directory to index and explore.")

    args = parser.parse_args()
    directory = os.path.abspath(os.path.expanduser(args.directory))
    collection = collection_name(directory)
    history_file = os.path.join(os.getenv("HOME"), ".explore", f"history-{collection}")
    with open(history_file, "a"):
        pass

    readline.read_history_file(history_file)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_store = Chroma(
        collection_name=collection,
        embedding_function=embedding_model,
        # TODO: need to figure out doc/chunk deduplication and upsert behavior before enabling persistence
        #        persist_directory=os.path.join(os.getenv("HOME"), ".explore", "db-langchain"),
    )

    docs = collect_documents(directory)
    vector_store.add_documents(docs)

    # llm = ChatOllama(model="mistral-nemo:latest")
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant who answers questions about a codebase. Use the following pieces of retrieved context from the codebase to answer the question. If you don't know the answer, just say that you don't know. Keep your answers concise and to the point. Make sure to mention the specific files and code from the context that you are referring to in your answers.\n\n{context}",
            ),
            ("human", "{input}"),
        ]
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    # potential improvement here: use ParentDocumentRetriever to retrieve full or larger-chunk docs given smaller child chunks that are indexed
    # https://python.langchain.com/docs/how_to/parent_document_retriever/
    # Other interesting strategies here: https://python.langchain.com/docs/how_to/multi_vector/
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_retriever, llm=llm
    )
    qa_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_prompt=PromptTemplate.from_template("{source}:\n{page_content}"),
    )
    retrieval_chain = create_retrieval_chain(
        retriever=multi_query_retriever, combine_docs_chain=qa_chain
    )
    question = input("Ask a question about the codebase: ")
    readline.write_history_file(history_file)

    # The rich library can't parse streaming Markdown, so we can't stream if we want Markdown rendering
    with console.status("Thinking..."):
        response = retrieval_chain.invoke({"input": question})
    md = Markdown(response["answer"])
    console.print(md)


if __name__ == "__main__":
    main()
