import argparse
import os

import pandas as pd
import torch
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_relevancy, faithfulness
from ragas.run_config import RunConfig

# Disable usage tracking for Ragas to ensure data privacy.
os.environ['RAGAS_DO_NOT_TRACK'] = 'true'


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

    # Convert the DataFrame to a list of dictionaries for compatibility with `datasets.Dataset`.
    data_dict_list = df.to_dict("records")
    for data_dict in data_dict_list:
        data_dict["contexts"] = [data_dict["context"]]
        del data_dict["context"]
    dataset = Dataset.from_list(data_dict_list)

    # Initialize the OpenAI Chat model with temperature set to 0 for deterministic outputs.
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
        ],
        raise_exceptions=False,
        is_async=False,
        run_config=RunConfig(
            timeout=60,
            max_workers=1
        )
    )

    df = result.to_pandas()

    # Reorder columns for better readability.
    df = df[["question", "answer", "faithfulness", "answer_relevancy",
             "context_relevancy", "ground_truth", "contexts"]]

    print(df[["faithfulness", "answer_relevancy", "context_relevancy"]])
    df.to_csv(f"{output_path}/eval_result.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_path", required=True, help="Directory to store evaluation results.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    args = parser.parse_args()
    run(args.input_path, args.output_path, args.api_key)
