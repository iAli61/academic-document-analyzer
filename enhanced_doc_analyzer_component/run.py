import argparse
from pathlib import Path
import pandas as pd
from enhanced_document_analyzer import EnhancedDocumentAnalyzer

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Input arguments
    parser.add_argument("--input_pdf", type=str, required=True)
    parser.add_argument("--azure_api_key", type=str, required=True)
    parser.add_argument("--azure_endpoint", type=str, required=True)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--overlap_threshold", type=float, default=0.5)
    parser.add_argument("--ignore_roles", type=str, default="pageFooter,footnote")
    
    # Output arguments
    parser.add_argument("--markdown_output", type=str, required=True)
    parser.add_argument("--elements_data", type=str, required=True)
    parser.add_argument("--visualizations", type=str, required=True)
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_dir = Path(args.visualizations)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = EnhancedDocumentAnalyzer(
        api_key=args.azure_api_key,
        endpoint=args.azure_endpoint,
        output_dir=str(output_dir),
        confidence_threshold=args.confidence_threshold,
        min_length=args.min_length,
        overlap_threshold=args.overlap_threshold,
        ignor_roles=args.ignore_roles.split(",")
    )
    
    try:
        # Process document
        markdown_text, elements_df, visualizations = analyzer.analyze_document(args.input_pdf)
        
        # Save markdown output
        with open(args.markdown_output, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        
        # Save elements data
        elements_df.to_csv(args.elements_data, index=False)
        
        # Copy visualizations to output directory
        import shutil
        for page_num, vis_path in visualizations.items():
            src_path = Path(vis_path)
            dst_path = vis_dir / src_path.name
            shutil.copy2(src_path, dst_path)
        
        print(f"Processing complete!")
        print(f"Total elements detected: {len(elements_df)}")
        print(f"Visualization pages generated: {len(visualizations)}")
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise

if __name__ == "__main__":
    main()