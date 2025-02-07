#!/usr/bin/env python
# run.py

import argparse
import logging
import traceback
from openai import AzureOpenAI
from azureml.rag.utils.connections import get_connection_by_id_v2
from azureml.rag.utils.logging import get_logger, safe_mlflow_start_run, track_activity

from document_processor import DocumentProcessor

logger = get_logger("document_analyzer")

def setup_openai_client(connection_id: str) -> tuple:
    """Set up Azure OpenAI client using connection."""
    connection = get_connection_by_id_v2(connection_id)
    endpoint = connection['properties']['target']
    api_key = connection['properties']['credentials']['keys']['key']
    
    # Create clients for vision and embeddings
    vision_client = AzureOpenAI(
        api_key=api_key,
        api_version="2023-03-15-preview",
        azure_endpoint=endpoint
    )
    
    embedding_client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=endpoint
    )
    
    return vision_client, embedding_client

def setup_search_connection(connection_id: str) -> tuple:
    """Set up Azure Search connection details."""
    connection = get_connection_by_id_v2(connection_id)
    endpoint = connection['properties']['target']
    api_key = connection['properties']['credentials']['keys']['api_key']
    api_version = connection['properties'].get('metadata', {}).get('apiVersion', "2023-07-01-preview")
    
    return endpoint, api_key, api_version

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Input arguments
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--azure_openai_connection_id", type=str, required=True)
    parser.add_argument("--azure_search_connection_id", type=str, required=True)
    parser.add_argument("--embd_deployment_name", type=str, required=True)
    parser.add_argument("--vision_deployment_name", type=str, required=True)
    parser.add_argument("--index_name", type=str, required=True)
    parser.add_argument("--max_chunk_length", type=int, default=4000)
    parser.add_argument("--output_folder", type=str, required=True)
    
    return parser.parse_args()

def main(args, logger):
    """Main function to set up and run document processing."""
    try:
        # Set up Azure OpenAI clients
        vision_client, embedding_client = setup_openai_client(args.azure_openai_connection_id)
        
        # Set up Azure Search connection
        search_endpoint, search_key, search_api_version = setup_search_connection(
            args.azure_search_connection_id
        )
        
        # Initialize document processor
        processor = DocumentProcessor(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            openai_client=vision_client,
            vision_deployment_name=args.vision_deployment_name,
            embedding_client=embedding_client,
            embd_deployment_name=args.embd_deployment_name,
            search_endpoint=search_endpoint,
            search_key=search_key,
            search_api_version=search_api_version,
            index_name=args.index_name,
            max_chunk_length=args.max_chunk_length
        )
        
        # Run processing
        stats, _ = processor.process()
        logger.info(f"Processing completed with stats: {stats}")
        
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main_wrapper(args):
    """Wrapper function to handle logging and MLflow run context."""
    with track_activity(logger, "document_analyzer") as activity_logger, \
         safe_mlflow_start_run(logger=logger):
        try:
            main(args, activity_logger)
        except Exception:
            activity_logger.error(
                f"document_analyzer failed with exception: {traceback.format_exc()}"
            )
            raise

if __name__ == "__main__":
    args = parse_args()
    main_wrapper(args)