import argparse
from fnmatch import fnmatch
import os
from langchain.text_splitter import Language
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPattern
from sklearn.base import defaultdict

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
        # TODO: need to figure out doc/chunk deduplication and upsert behavior before enabling persistence
        #        persist_directory=os.path.join(os.getenv("HOME"), ".explore", "db-langchain"),
    )

    docs = collect_documents(directory)
    vector_store.add_documents(docs)

    llm = ChatOllama(model="mistral-nemo:latest")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant who answers questions about a codebase. Use the following pieces of retrieved context from the codebase to answer the question. If you don't know the answer, just say that you don't know. Keep your answers concise and to the point.",
            ),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    question = input("Ask a question about the codebase: ")
    response = chain.invoke(question)

    print(response)

    # create chroma store from persisted path
    # index any changed/new documents
    # create chat engine
    # RAG: have the LLM generate a query string to send to the vector store to retrieve relevant docs, then
    # have the LLM generate a response given the query and context
    # run chat in loop


if __name__ == "__main__":
    main()
