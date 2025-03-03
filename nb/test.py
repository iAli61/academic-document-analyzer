
import sys
sys.path.append('/app/enhanced_doc_analyzer_component/')

from enhanced_document_analyzer import EnhancedDocumentAnalyzer
import dotenv
import os
import pandas as pd

# Load the environment variables
dotenv.load_dotenv()

# Initialize the analyzer
ct = 0.55
pdf_path = "/app/nb/files/2412.20995v1.pdf"

output_dir = f"output-azure/{ct}/{os.path.basename(pdf_path)}"
analyzer = EnhancedDocumentAnalyzer(
        api_key=os.getenv("DOCUMENT_INTELLIGENCE_KEY"),
        endpoint=os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT"),
        output_dir=output_dir,
        confidence_threshold=0.5,
        overlap_threshold=0.3  # More aggressive overlap detection
    )

markdown_text, elements_df, visualization_paths  = analyzer.analyze_document(pdf_path)

# Print the markdown text
print(markdown_text)



