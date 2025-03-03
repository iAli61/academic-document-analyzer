# Document Processing Pipeline

## Overview
This Azure Machine Learning pipeline combines document analysis and content indexing capabilities in a two-stage process:
1. Document Analysis: Processes PDF documents using Azure Document Intelligence and local layout detection
2. Chunk & Index: Splits processed documents into chunks and indexes them with Azure Cognitive Search

## Architecture

### Components

#### 1. Document Analyzer
- Processes PDF documents using Azure Document Intelligence
- Performs layout detection using a deep learning model
- Extracts text, tables, and figures
- Generates visualizations of detected elements
- Outputs structured data and element images

#### 2. Chunk Caption Index
- Takes processed documents from Document Analyzer
- Splits content into semantic chunks
- Generates captions for images
- Creates search indexes in Azure Cognitive Search

### Required Azure Services
- Azure Machine Learning
- Azure Document Intelligence
- Azure OpenAI
- Azure Cognitive Search

## Prerequisites

### Azure Resources
1. Azure Machine Learning workspace
2. Azure Document Intelligence service
3. Azure OpenAI service with the following deployments:
   - text-embedding-ada-002
   - gpt-4o (or specified vision model)
4. Azure Cognitive Search service

### Azure ML Connections
Create the following connections in your Azure ML workspace:
1. Document Intelligence connection
2. Azure OpenAI connection
3. Azure Cognitive Search connection

### Environment Setup
1. Install the required environment using the provided conda.yaml:
```bash
conda env create -f conda.yaml
```

2. Register the environment in Azure ML:
```bash
az ml environment create --file env.yaml
```

## Pipeline Components

### Document Analyzer Component
Parameters:
- `input_folder`: Input folder containing PDF files
- `doc_intel_connection_id`: Azure ML connection ID for Document Intelligence
- `confidence_threshold`: Confidence threshold for element detection (default: 0.7)
- `min_length`: Minimum text length to consider (default: 10)
- `overlap_threshold`: Threshold for overlap detection (default: 0.5)
- `ignore_roles`: Comma-separated list of roles to ignore (default: "pageFooter,footnote")

### Chunk Caption Index Component
Parameters:
- `input_folder`: Input folder from Document Analyzer output
- `azure_openai_connection_id`: Azure ML connection ID for OpenAI service
- `azure_search_connection_id`: Azure ML connection ID for Azure Search service
- `embd_deployment_name`: Azure OpenAI embeddings model deployment name
- `vision_deployment_name`: Azure OpenAI GPT-4V model deployment name
- `index_name`: Azure Search index name
- `max_chunk_length`: Maximum length of text chunks (default: 4000)

## Usage

### Running the Pipeline

```python
from azure.ai.ml import dsl, Input
from azure.ai.ml import load_component, load_environment

# Load components and environment
analyzer_component = load_component(source="./doc_analyzer_component.yaml")
chunk_caption_index = load_component(source="./chunk_caption_index_component/chunk-caption-index-component.yaml")
environment = load_environment(source="./env.yaml")

# Get your connections
doc_intelligence_connection = ml_client.connections.get("my-doc-intelligence-connection")
azure_search_connection = ml_client.connections.get("aisearch505")
azure_openai_connection = ml_client.connections.get("aoai-sweden-505")

# Create and run pipeline
pipeline = document_processing_pipeline(
    # Document Analyzer params
    pdf_folder="azureml:raw_papers:1",
    doc_intel_connection_id=doc_intelligence_connection.id,
    confidence_threshold=0.3,
    min_length=15,
    overlap_threshold=0.7,
    ignore_roles="pageFooter,footnote,pageHeader",
    
    # Chunk Caption Index params
    azure_openai_connection_id=azure_openai_connection.id,
    azure_search_connection_id=azure_search_connection.id,
    embd_deployment_name="text-embedding-ada-002",
    vision_deployment_name="gpt-4v",
    index_name="myindex",
    
    # Compute settings
    analyzer_compute="hp-gpu-cluster",
    indexer_compute="cpu-cluster"
)

# Submit the pipeline
pipeline_job = ml_client.jobs.create_or_update(pipeline)
```

## Output Structure

The pipeline produces the following outputs:

### Document Analyzer Output
```
output_dir/
├── {pdf_name}/
│   ├── analysis.md
│   ├── elements.csv
│   ├── elements/
│   │   ├── text/
│   │   ├── tables/
│   │   └── figures/
│   └── visualizations/
|       ├── page_{page_number}.png
│       └── overlay/
|           └── {element_name}.png
```

## Monitoring and Logging

The pipeline provides comprehensive logging and monitoring capabilities:

1. Azure ML metrics tracking:
   - Number of processed files
   - Elements detected per file
   - Total elements processed
   - Processing time per stage

2. Output logs:
   - Processing status for each file
   - Error logging
   - Performance metrics

Access logs and metrics through:
- Azure ML Studio UI
- Azure ML SDK
- Azure ML CLI

