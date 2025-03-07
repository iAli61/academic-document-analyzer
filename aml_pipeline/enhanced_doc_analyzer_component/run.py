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
    parser.add_argument("--top_margin_percent", type=float, default=0.05)
    parser.add_argument("--bottom_margin_percent", type=float, default=0.05)
    parser.add_argument("--ocr_elements", type=str, default="formula,table")
    
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
        # Initialize combined report data
        combined_report = {
            "documents": [],
            "total_documents": 0,
            "total_pages": 0,
            "total_time_seconds": 0,
            "total_nougat_images": 0
        }
        
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
                ignor_roles=args.ignore_roles.split(","),
                top_margin_percent=args.top_margin_percent,
                bottom_margin_percent=args.bottom_margin_percent,
                ocr_elements=args.ocr_elements.split(",")
            )
            logger.info(f"Processing {pdf_file.name}")
            try:
                pdf = process_single_pdf(pdf_file, analyzer, logger)
                if not pdf.empty:
                    all_dfs.append(pdf)
                
                # Collect report data from this analyzer
                combined_report["documents"].extend(analyzer.report_data["documents"])
                combined_report["total_documents"] += analyzer.report_data["total_documents"]
                combined_report["total_pages"] += analyzer.report_data["total_pages"]
                combined_report["total_time_seconds"] += analyzer.report_data["total_time_seconds"]
                combined_report["total_nougat_images"] += analyzer.report_data["total_nougat_images"]
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Combine all DataFrames
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_csv(f"{args.output_dir}/combined_elements_data.csv", index=False)
            logger.info(f"Combined data saved to {args.output_dir}/combined_elements_data.csv")
            logger.info(f"Total elements across all PDFs: {len(combined_df)}")
        else:
            logger.warning("No data was successfully processed")
        
        # Format the combined report
        if combined_report["total_documents"] > 0:
            # Format time and add aggregate metrics
            for doc in combined_report["documents"]:
                if "processing_time_seconds" in doc:
                    doc["processing_time_formatted"] = _format_time(doc["processing_time_seconds"])
            
            combined_report["total_time_formatted"] = _format_time(combined_report["total_time_seconds"])
            
            # Calculate averages
            combined_report["avg_pages_per_document"] = combined_report["total_pages"] / combined_report["total_documents"]
            combined_report["avg_processing_time_per_document"] = combined_report["total_time_seconds"] / combined_report["total_documents"]
            combined_report["avg_nougat_images_per_document"] = combined_report["total_nougat_images"] / combined_report["total_documents"]
            if combined_report["total_pages"] > 0:
                combined_report["avg_processing_time_per_page"] = combined_report["total_time_seconds"] / combined_report["total_pages"]
            
            # Add timestamp
            combined_report["report_generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Write combined report
            combined_report_path = Path(args.output_dir) / "combined_report.json"
            with open(combined_report_path, 'w') as f:
                json.dump(combined_report, f, indent=4)
            
            logger.info(f"Combined report saved to {combined_report_path}")
            logger.info(f"Total documents processed: {combined_report['total_documents']}")
            logger.info(f"Total pages processed: {combined_report['total_pages']}")
            logger.info(f"Total Nougat images: {combined_report['total_nougat_images']}")
            logger.info(f"Total processing time: {combined_report['total_time_formatted']}")
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Add this helper function to format time consistently
def _format_time(seconds):
    """Format time in seconds to a human-readable string."""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"

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