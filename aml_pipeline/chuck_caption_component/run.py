#!/usr/bin/env python
import argparse
import logging
import traceback
from openai import AzureOpenAI
from azureml.rag.utils.connections import get_connection_by_id_v2
from azureml.rag.utils.logging import get_logger, safe_mlflow_start_run, track_activity
import os

from src import DocumentProcessor

logger = get_logger("document_analyzer")

def setup_openai_client(connection_id: str) -> AzureOpenAI:
    """Set up Azure OpenAI client using connection."""
    try:
        # Get connection details
        connection = get_connection_by_id_v2(connection_id)
        
        # Access properties using object methods/attributes
        endpoint = connection.endpoint
        api_key = connection.api_key
        
        # Create client for vision
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2023-03-15-preview",
            azure_endpoint=endpoint
        )
        
        return client
        
    except Exception as e:
        logger.error(f"Failed to setup Azure OpenAI connection: {str(e)}")
        raise

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Input arguments
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--azure_openai_connection_id", type=str, required=True)
    parser.add_argument("--vision_deployment_name", type=str, required=True)
    parser.add_argument("--max_chunk_length", type=int, default=4000)
    parser.add_argument("--output_folder", type=str, required=True)
    
    return parser.parse_args()

def main(args, logger):
    """Main function to set up and run document processing."""
    try:
        # Set up Azure OpenAI client
        vision_client = setup_openai_client(args.azure_openai_connection_id)

        logger.info(f"input_folder: {args.input_folder}")
        # print the list of files in the input folder
        
        for root, dirs, files in os.walk(args.input_folder):
            for file in files:
                logger.info(f"file: {file}")
                print(f"file: {file}")
        logger.info(f"output_folder: {args.output_folder}")
        
        # Initialize document processor
        processor = DocumentProcessor(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            openai_client=vision_client,
            vision_deployment_name=args.vision_deployment_name,
            max_chunk_length=args.max_chunk_length
        )
        
        # Run processing
        stats, output_file = processor.process()
        logger.info(f"Processing completed with stats: {stats}")
        if output_file:
            logger.info(f"Output saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main_wrapper(args):
    """Wrapper function to handle logging and MLflow run context."""
    
    try:
        main(args, logger)
    except Exception:
        logger.error(
            f"document_analyzer failed with exception: {traceback.format_exc()}"
        )
        raise

if __name__ == "__main__":
    args = parse_args()
    main_wrapper(args)