# 20 Minuten - Sentimental & Topic Analysis

This project performs topic modeling and sentiment analysis on a collection of articles. The main script, `main.py`,
orchestrates the entire pipeline, including data preprocessing, sentiment analysis, and topic modeling using various
techniques such as Top2Vec, LSA, and LDA.

## Requirements

- Python 3.11
- Required Python packages (install using `pip install -r requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ETS-HS24/20-minuten.git
    cd 20-minuten
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv .venv
    source .venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the main script, use the following command:

```sh
  python main.py
```

# Topic Modeling and Sentiment Analysis Pipeline

This project is a comprehensive text-processing pipeline that includes file handling, topic modeling, sentiment analysis, and multilingual topic matching. The main components include reading and preprocessing text data, performing sentiment analysis, and extracting topics through Top2Vec, LSA, and LDA methods. The pipeline processes multilingual data in French and German, performing topic matching across languages.

## Table of Contents
- [Project Structure](#project-structure)
- [Features](#features)
- [Dependencies](#dependencies)
- [Pipeline Stages](#pipeline-stages)
- [Configuration Options](#configuration-options)

## Project Structure and Features

The pipeline structure of the project encourages an explorative data analysis. Each step (file reading, text preprocessing, etc.) can be saved as an intermediate step. This enables the user to inspect the data at each stage and resume processing from any point in the pipeline.

The project is structured modularly, with each service handling a specific task in the text-processing pipeline. The key components and features are as follows:

- **FileService**: Handles file input and output operations, including reading and writing in formats such as `.tsv`, `.parquet`, and `.csv`.
- **TextService**: Provides text preprocessing utilities to clean the data, including lemmatization, tag processing, and column management, while filtering articles based on a specified character length.
- **SentimentService**: Conducts sentiment analysis, adding sentiment scores to each article.
- **TopicModelingService**: Performs topic extraction through three main techniques:
  - **Top2Vec**: Uses UMAP for dimensionality reduction and HDBSCAN for clustering to generate topics from text data.
  - **LSA and LDA**: Applies Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA) algorithms to identify the main topics and associated keywords.
- **TopicMatcherService**: Matches topics across languages using both translation and sentence transformer-based similarity, enabling cross-language topic matching between French and German articles.

Each component contributes to the overall pipeline, which reads raw text files, preprocesses and cleans the data, performs sentiment analysis, and extracts and matches topics, providing detailed insights across multilingual datasets.


## Configuration Options
The pipeline allows configuration of key parameters:
- `force_recreate`: Force recreation of all intermediate datasets and models.
- `article_length_threshold`: Minimum character count for articles to be included in processing.
- **Top2Vec Model Configurations**:
  - `n_neighbors`, `n_components`, `metric`: Parameters for UMAP dimensionality reduction.
  - `min_cluster_size`, `metric`, `cluster_selection_method`: Parameters for HDBSCAN clustering.
- **LSA & LDA Parameters**:
  - `number_of_topics`: Number of topics for LSA and LDA models.
  - `number_of_top_words`: Number of top words for each topic.
  - `match_score`: Minimum similarity score for topic matching.

## Logging
Logs are outputted to `stdout` to track each stage's status, which is useful for debugging and monitoring.