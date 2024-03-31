import argparse
from operator import itemgetter

import pandas as pd
import torch
import weaviate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from weaviate.embedded import EmbeddedOptions


def get_llm_model(model_name, api_key, seed=42):
    model = None
    if model_name.startswith("gpt"):
        model = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=0,
            streaming=False,
            model_kwargs={"seed": seed}
        )
    elif model_name.startswith("gemini"):
        model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
            max_tokens=16384,
            convert_system_message_to_human=True
        )
    else:
        raise ValueError("Unsupported model_id: {model_id}")
    return model


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


def get_retriever(index_path, index_name, embedding_model, top_k=3):
    client = weaviate.Client(
        embedded_options=EmbeddedOptions(
            persistence_data_path=index_path
        )
    )

    vectorstore = Weaviate(
        embedding=embedding_model,
        client=client,
        index_name=index_name,
        text_key="text",
        by_text=False,
        attributes=["source", "title"]
    )
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def format_docs(docs):
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = (
            f"<document index='{i}'>\n"
            f"<source>{doc.metadata.get('source')}</source>\n"
            f"<doc_content>{doc.page_content}</doc_content>\n"
            "</document>"
        )
        formatted_docs.append(doc_string)
    formatted_str = "\n".join(formatted_docs)
    return f"<documents>\n{formatted_str}\n</documents>"


def run(input_path, output_path, index_path, index_name, model_name, api_key, seed=42):

    llm_model = get_llm_model(model_name, api_key, seed=seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = get_embeddings_model(device)
    retriever = get_retriever(index_path, index_name, embedding_model)

    retrieve_chain = (
        RunnableAssign({
            "context": (itemgetter("question") | retriever | format_docs).with_config(
                run_name="FormatDocs"
            )
        })
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant answering questions about LangChain."
                "\n{context}\n"
                "Respond solely based on the document content.",
            ),
            ("human", "{question}")
        ]
    )

    response_generator = (prompt | llm_model | StrOutputParser()).with_config(
        run_name="GenerateResponse"
    )

    chain = retrieve_chain | response_generator

    df = pd.read_csv(input_path)
    data_dict_list = [{
        'question': row['question'], 'ground_truth': row['answer']
    } for _, row in df.iterrows()
    ]

    for data_dict in tqdm(data_dict_list):
        prompt_with_context = retrieve_chain.invoke(
            {'question': data_dict['question']})
        response = chain.invoke(prompt_with_context)
        data_dict['context'] = prompt_with_context['context']
        data_dict['answer'] = response

    result_df = pd.DataFrame(data_dict_list)
    result_df.to_csv(f'{output_path}/pred_{model_name}.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--index_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--api_key", required=True)
    args = parser.parse_args()

    run(args.input_path, args.output_path, args.index_path, args.index_name,
        args.model_name, args.api_key)
