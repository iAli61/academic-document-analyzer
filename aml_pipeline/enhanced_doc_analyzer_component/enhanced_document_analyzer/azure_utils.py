# azure_utils.py

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict
from .document_element_record import DocumentElementRecord, BoundingBox
from .document_element_type import DocumentElementType
import json
import os
from pathlib import Path
import pymupdf

def analyze_page_with_azure(azure_client: DocumentAnalysisClient, pdf_path: str, page_num: int, output_dir: str) -> dict:
    """Analyze a single page with Azure Document Intelligence."""
    output_dir = Path(output_dir)
    azure_cache_dir = output_dir / "azure_cache"
    azure_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if the analysis result already exists for this page
    cache_file = azure_cache_dir / f"page_{page_num}_azure_result.json"
    if cache_file.exists():
        print(f"Loading cached analysis for page {page_num}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    try:
        # Extract single page to temporary PDF
        doc = pymupdf.open(pdf_path)
        new_doc = pymupdf.open()  # Create new PDF
        new_doc.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
        
        temp_pdf = azure_cache_dir / f"temp_page_{page_num}.pdf"
        new_doc.save(str(temp_pdf))
        doc.close()
        new_doc.close()

        # Analyze single page with Azure
        with open(temp_pdf, "rb") as f:
            poller = azure_client.begin_analyze_document("prebuilt-document", document=f)
            result = poller.result().to_dict()

        # Cache the result
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        # Clean up temporary PDF
        temp_pdf.unlink()

        return result

    except Exception as e:
        print(f"Error analyzing page {page_num}: {str(e)}")
        return {
            'pages': [{
                'page_number': page_num,
                'width': 8.5,  # Standard letter size in inches
                'height': 11,
                'unit': 'inch',
                'tables': [],
                'paragraphs': []
            }]
        }

def process_azure_paragraphs(paragraphs: List[Dict], pdf_name: str, page_info: Dict, page_num: int, ignor_roles: List[str], min_length: int) -> List[DocumentElementRecord]:
    """Process text paragraphs from Azure Document Intelligence."""
    elements = []
    
    # Store page dimensions
    page_width = float(page_info['width'])
    page_height = float(page_info['height'])
    page_unit = page_info['unit']
    
    # Add order_id to track original Azure ordering
    for order_id, para in enumerate(paragraphs):
        if para['role'] in ignor_roles:
            continue
        if len(para['content']) < min_length:
            continue
            
        if para['bounding_regions'][0]['page_number'] == page_num:
            elements.append(DocumentElementRecord(
                pdf_file=pdf_name,
                page=page_num,
                bounding_box=BoundingBox.from_azure_regions(para['bounding_regions']),
                element_type=DocumentElementType.TEXT,
                text=para['content'],
                role=para['role'],
                spans=para['spans'],
                metadata={
                    'source': 'azure_document_intelligence',
                    'page_width': page_width,
                    'page_height': page_height,
                    'page_unit': page_unit,
                    'azure_order_id': order_id
                }
            ))
    
    return elements