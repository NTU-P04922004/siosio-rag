# siosio-rag
This open-source project demonstrates Retrieval Augmented Generation (RAG) using Large Language Models (LLMs).

## Overview
**siosio-rag** uses the following technologies:
* **Orchestration:** LangChain.
* **Embedding Vectorstore:** Weaviate.
* **Text Embeddings:** FlagEmbedding.
* **LLMs:** OpenAI ChatGPT and Google Gemini.
* **Evaluation:** Ragas.

## Getting Started

### Prerequisites

- `Python` (version 3.10 or higher)
- `PyTorch` (version 2.1.2 or higher)

Once you have these dependencies installed, clone this repository and set up the environment:

### Installation

#### Clone the repository

```bash
git clone https://github.com/your-username/siosio-rag.git
```

#### Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Prepare Data
Run `prepare_data.py` to preprocess and save the data:

```bash
python prepare_data.py --out_path target_directory_path
```

**Note:** Replace `target_directory_path` with the desired directory for storing the processed data.

### Build Index for Retrieval
Run `build_index.py` to convert data into embeddings and build an index:

```bash
python build_index.py --input_path data_directory_path --out_path target_directory_path --index_name name_for_index
```

**Note:**
* Replace `data_directory_path` with the directory containing the processed data.
* Replace `target_directory_path` with the desired directory for storing the index.
* Replace `name_for_index` with a chosen name for your index.

### Perform RAG
Run `rag.py` to execute RAG:

```bash
python rag.py \
    --index_path index_directory_path \
    --index_name name_for_index \
    --model_name llm_model_name \
    --api_key api_key_for_llm \
    --query query_text
```

**Note:**
* Replace `index_directory_path` with the directory containing the index.
* Replace `name_for_index` with the name you used for your index.
* Replace `llm_model_name` with the desired LLM model (e.g., "gpt-3.5-turbo" or "gemini-1.0-pro-001").
* Replace `api_key_for_llm` with your API key for the chosen LLM.
* Replace `query_text` with your desired query for the RAG process.

### Evaluation (Optional)
Run `evaluate.py` to evaluate the RAG results:

```bash
python evaluate.py \
    --input_path result_csv_path \
    --output_path target_directory_path \
    --api_key api_key_for_llm
```

**Note:**
* Replace `result_csv_path` with the path of the result csv file.
* Replace `target_directory_path` with the desired directory for saving the evaluation result.
* Replace `api_key_for_llm` with your API key for OpenAI.

## License
This project is licensed under the Apache-2.0 license. Please refer to the LICENSE file for details.