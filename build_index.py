import argparse
import json

import torch
import weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_core.documents import Document
from weaviate.embedded import EmbeddedOptions


def load_docs_from_jsonl(file_path):
    docs = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            obj = Document(**data)
            docs.append(obj)
    return docs


def get_embeddings_model(device):
    model_name = "BAAI/bge-base-en-v1.5"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=False
    )
    return embedding_model


def run(input_path, out_path, index_name, chunk_size, chunk_overlap):
    docs_from_documentation = load_docs_from_jsonl(f"{input_path}/langchain_docs.json")
    docs_from_langsmith = load_docs_from_jsonl(
        f"{input_path}/langsmith_docs.json")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs_transformed = text_splitter.split_documents(
        docs_from_documentation + docs_from_langsmith
    )
    docs_transformed = [
        doc for doc in docs_transformed if len(doc.page_content) > 10]

    # Set empty "source" and "title" metadata when they do not exsit.
    # (Weaviate will error if one of the attributes is missing from a document.)
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    client = weaviate.Client(
        embedded_options=EmbeddedOptions(
            persistence_data_path=out_path
        )
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = get_embeddings_model(device)

    # Save embedding data to the vectorstore.
    Weaviate.from_documents(
        docs_transformed,
        embedding_model,
        client=client,
        index_name=index_name,
        text_key="text",
        by_text=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Directory path that contains the resource doc.")
    parser.add_argument("--out_path", required=True, help="Directory to store results.")
    parser.add_argument("--index_name", default="Langsmith_docs_test", help="Index name for the vectorstore.")
    parser.add_argument("--chunk_size", type=int, default=4000, help="Chunk size to split the input text.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap size between text chunks.")
    args = parser.parse_args()
    run(args.input_path, args.out_path, args.index_name, args.chunk_size, args.chunk_overlap)
