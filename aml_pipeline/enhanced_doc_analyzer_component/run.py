import argparse
from pathlib import Path
import pandas as pd
import os
import traceback
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from enhanced_document_analyzer import EnhancedDocumentAnalyzer
from azureml.rag.utils.connections import get_connection_by_id_v2
from azureml.rag.utils.logging import get_logger, safe_mlflow_start_run, track_activity
import glob

logger = get_logger("document_analyzer")

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Input arguments
    parser.add_argument("--input_folder", type=str, required=True,
                       help="Input folder containing PDF files")
    parser.add_argument("--doc_intel_connection_id", type=str, required=True,
                       help="Azure ML connection ID for Document Intelligence")
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--overlap_threshold", type=float, default=0.5)
    parser.add_argument("--ignore_roles", type=str, default="pageFooter,footnote")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True)
    
    return parser.parse_args()

def setup_document_intelligence(connection_id: str) -> tuple:
    """
    Set up Document Intelligence client using Azure ML connection.
    
    Args:
        connection_id: Azure ML connection ID for Document Intelligence
        
    Returns:
        tuple: (endpoint, api_key)
    """
    try:
        # Get connection details
        doc_intelligence_connection = get_connection_by_id_v2(connection_id)
        
        # Extract credentials
        endpoint = doc_intelligence_connection["properties"]["metadata"]["endpoint"]
        api_key = doc_intelligence_connection["properties"]["credentials"]["keys"]["api_key"]
        
        # Set environment variables
        os.environ["DOCUMENT_INTELLIGENCE_ENDPOINT"] = endpoint
        os.environ["DOCUMENT_INTELLIGENCE_KEY"] = api_key
        
        return endpoint, api_key
        
    except Exception as e:
        logger.error(f"Failed to setup Document Intelligence connection: {str(e)}")
        raise

def process_single_pdf(pdf_path: Path, analyzer: EnhancedDocumentAnalyzer, logger) -> pd.DataFrame:
    """
    Process a single PDF file and return its elements DataFrame.
    
    Args:
        pdf_path: Path to the PDF file
        analyzer: Initialized EnhancedDocumentAnalyzer instance
        markdown_dir: Directory for markdown output
        vis_dir: Directory for visualizations
        logger: Logger instance
        
    Returns:
        pd.DataFrame: Elements data for this PDF
    """
    try:
        # Process document
        markdown_text, elements_df, visualizations = analyzer.analyze_document(str(pdf_path))
        
        # Save markdown output
        markdown_file = f"{analyzer.output_dir}/{pdf_path.stem}_analysis.md"
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        
        # Add filename column to DataFrame
        elements_df['source_pdf'] = pdf_path.name

        # Save results
        elements_df.to_csv(f"{analyzer.output_dir}/{pdf_path.stem}_elements.csv", index=False)
        
        logger.info(f"Successfully processed {pdf_path.name}")
        logger.info(f"Elements detected: {len(elements_df)}")
        logger.info(f"Visualizations generated: {len(visualizations)}")
        
        return elements_df
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()  # Return empty DataFrame on error

def main(args, logger):
    """
    Main function to run document analysis.
    
    Args:
        args: Parsed command line arguments
        logger: Logger instance
    """
    try:

        # Set up Document Intelligence credentials
        endpoint, api_key = setup_document_intelligence(args.doc_intel_connection_id)
        
        
        # Get list of PDF files
        input_folder = Path(args.input_folder)
        pdf_files = list(input_folder.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_folder}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF and collect DataFrames
        all_dfs = []
        for pdf_file in pdf_files:

            output_dir = Path(args.output_dir) / pdf_file.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            # Initialize analyzer
            analyzer = EnhancedDocumentAnalyzer(
                api_key=api_key,
                endpoint=endpoint,
                output_dir=str(output_dir),
                confidence_threshold=args.confidence_threshold,
                min_length=args.min_length,
                overlap_threshold=args.overlap_threshold,
                ignor_roles=args.ignore_roles.split(",")
            )
            logger.info(f"Processing {pdf_file.name}")
            try:
                pdf = process_single_pdf(pdf_file, analyzer, logger)
                if not pdf.empty:
                    all_dfs.append(pdf)
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Combine all DataFrames
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_csv(f"{args.output_dir}/combined_elements_data.csv", index=False)
            logger.info(f"Combined data saved to {args.output_dir}.combined_elements_data.csv")
            logger.info(f"Total elements across all PDFs: {len(combined_df)}")
        else:
            logger.warning("No data was successfully processed")
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main_wrapper(args):
    """
    Wrapper function to handle logging and MLflow run context.
    
    Args:
        args: Parsed command line arguments
    """
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