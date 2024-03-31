import argparse

import pandas as pd
import torch
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_relevancy, faithfulness


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


def run(input_path, output_path, api_key):
    df = pd.read_csv(input_path)
    data_dict_list = df.to_dict("records")
    for d in data_dict_list:
        d["contexts"] = [d["context"]]
        del d["context"]
    dataset = Dataset.from_list(data_dict_list)

    openai_model = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=api_key,
        temperature=0,
        streaming=False,
        model_kwargs={"seed": 42}
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = evaluate(
        dataset,
        llm=openai_model,
        embeddings=get_embeddings_model(device),
        metrics=[
            faithfulness,
            answer_relevancy,
            context_relevancy
        ]
    )

    df = result.to_pandas()
    print(df[["faithfulness", "answer_relevancy", "context_relevancy"]])
    df.to_csv(f"{output_path}/eval_result.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--api_key", required=True)
    args = parser.parse_args()
    run(args.input_path, args.output_path, args.api_key)
