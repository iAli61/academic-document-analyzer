import pandas as pd
import pymupdf   # PyMuPDF
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict
import os
import numpy as np
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from .document_element_type import DocumentElementType
from .document_element_record import DocumentElementRecord, BoundingBox
from .layout_detection_service import LayoutDetectionService
from .nougat_service import NougatService
from .document_types import DocumentElement
from .bounding_box_visualizer import BoundingBoxVisualizer
from .bounding_box_scaler import BoundingBoxScaler
from .azure_utils import analyze_page_with_azure, process_azure_paragraphs
from .overlap_utils import filter_overlapping_elements
from .margin_detector import MarginDetector


class EnhancedDocumentAnalyzer:
    def __init__(self, 
                 api_key: str, 
                 endpoint: str, 
                 output_dir: str = "output",
                 confidence_threshold: float = 0.7,
                 min_length: int = 10,
                 overlap_threshold: float = 0.5,
                 ignor_roles: List[str] = ['pageFooter','footnote'],
                 top_margin_percent: float = 0.0,  # as a percentage of page height
                 bottom_margin_percent: float = 0.0
                 ):
        """Initialize the document analyzer with both Azure and local services."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Azure Document Intelligence client
        self.azure_client = DocumentAnalysisClient(endpoint, AzureKeyCredential(api_key))
        
        # Layout detection for images and tables
        self.layout_detector = LayoutDetectionService(confidence_threshold)
        
        # Initialize Nougat service for complex elements
        self.nougat_service = NougatService()
        
        self.min_length = min_length
        self.ignor_roles = ignor_roles
        self.overlap_threshold = overlap_threshold
        self.top_margin_percent = top_margin_percent
        self.bottom_margin_percent = bottom_margin_percent
        self.top_margin = 0
        self.bottom_margin = 0

    def _calculate_overlap(self, smaller_box: Tuple[float, float, float, float], 
                         larger_box: Tuple[float, float, float, float]) -> float:
        """
        Calculate the containment ratio of a smaller box within a larger box.
        
        Args:
            smaller_box: Smaller bounding box (x1, y1, x2, y2)
            larger_box: Larger bounding box (x1, y1, x2, y2)
            
        Returns:
            float: Containment ratio between 0 and 1
        """
        # Calculate intersection coordinates
        x1 = max(smaller_box[0], larger_box[0])
        y1 = max(smaller_box[1], larger_box[1])
        x2 = min(smaller_box[2], larger_box[2])
        y2 = min(smaller_box[3], larger_box[3])
        
        # Check if there is an intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate areas
        intersection = (x2 - x1) * (y2 - y1)
        smaller_area = (smaller_box[2] - smaller_box[0]) * (smaller_box[3] - smaller_box[1])
        
        # Calculate containment ratio (how much of the smaller box is contained in the larger box)
        return intersection / smaller_area if smaller_area > 0 else 0.0

    def _parse_box_string(self, box_str: str) -> Tuple[float, float, float, float]:
        """Parse a bounding box string into a tuple of coordinates."""
        return tuple(float(x.strip()) for x in box_str.strip('()').split(','))

    def _calculate_mutual_overlap(self, box1: Tuple[float, float, float, float], 
                                box2: Tuple[float, float, float, float]) -> float:
        """
        Calculate mutual overlap ratio between two boxes.
        
        Args:
            box1: First bounding box (x1, y1, x2, y2)
            box2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            float: Overlap ratio between 0 and 1
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Check if there is an intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate areas
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Use the minimum area for overlap ratio
        min_area = min(area1, area2)
        
        return intersection / min_area if min_area > 0 else 0.0
    
    def _get_azure_page_info(self, azure_result: dict, page_num: int) -> dict:
        """
        Safely get page information from Azure results with error handling.
        
        Args:
            self: Instance of EnhancedDocumentAnalyzer
            azure_result: Azure Document Intelligence results
            page_num: Page number to find
            
        Returns:
            dict: Page information or default values
        """
        try:
            # First try exact match
            for page in azure_result.get('pages', []):
                if page.get('page_number') == page_num:
                    return page
                    
            # If not found, try finding closest match
            available_pages = sorted([p.get('page_number') for p in azure_result.get('pages', []) 
                                if p.get('page_number') is not None])
            if not available_pages:
                raise ValueError(f"No valid pages found in Azure results")
                
            closest_page_num = min(available_pages, key=lambda x: abs(x - page_num))
            print(f"Warning: Page {page_num} not found, using closest match: {closest_page_num}")
            
            return next(p for p in azure_result['pages'] 
                    if p.get('page_number') == closest_page_num)
                    
        except Exception as e:
            print(f"Error getting page info for page {page_num}: {str(e)}")
            # Return default values as fallback
            return {
                'page_number': page_num,
                'width': 8.5,  # Standard letter size in inches
                'height': 11,
                'unit': 'inch',
                'tables': [],
                'paragraphs': []
            }
        
    # enhanced_document_analyzer.py

    def analyze_document(self, pdf_path: str) -> Tuple[str, pd.DataFrame, Dict[int, str]]:
        """
        Analyze a PDF document using multiple processing stages.
        
        Args:
            pdf_path: Path to the PDF file to analyze
            
        Returns:
            Tuple containing:
            - Markdown text representation of the document
            - DataFrame with element information
            - Dictionary mapping page numbers to visualization paths
        """
        pdf_path = Path(pdf_path)
        elements = []
        print(f"\nProcessing document: {pdf_path}")
        
        try:
            # Initialize bounding box scaler
            bbox_scaler = BoundingBoxScaler(str(pdf_path))
            
            # Step 1: Convert PDF pages to images first to know total pages
            print("\nStep 1: PDF to Image Conversion")
            images = self._pdf_to_images(pdf_path)
            print(f"Converting {len(images)} pages to images")

            # Calculate margin in pixels based on the first page's height
            if images:
                page_height = images[0].height
                self.top_margin = int(self.top_margin_percent * page_height)
                self.bottom_margin = int(self.bottom_margin_percent * page_height)
                print(f"Top Margin: {self.top_margin}, Bottom Margin: {self.bottom_margin}")
            
            # Step 2: Process each page
            print("\nStep 2: Page Processing")
            for page_num, page_img in enumerate(images, 1):
                try:
                    print(f"\nProcessing page {page_num}/{len(images)}")
                    
                    # 2a. Process with Azure Document Intelligence
                    print(f"Running Azure Document Analysis for page {page_num}")
                    azure_result = analyze_page_with_azure(
                        self.azure_client, 
                        pdf_path, 
                        page_num, 
                        self.output_dir
                    )
                    
                    # Update bounding box scaler with page dimensions
                    bbox_scaler.set_azure_dimensions(azure_result, page_num)
                    
                    # Get Azure page info with error handling
                    azure_page_info = self._get_azure_page_info(
                        azure_result=azure_result, 
                        page_num=1, #because we are using only one page
                        )
                    
                    # print(f"Azure page info: {azure_page_info}")
                    
                    # 2b. Detect layout elements
                    layout_elements = self.layout_detector.detect_elements(page_img, page_num)
                    print(f"Detected {len(layout_elements)} layout elements")
                    
                    # 2c. Save visualization of layout detection
                    vis_path = self.layout_detector.save_page_with_boxes(
                        page_img, layout_elements, self.output_dir, page_num
                    )
                    print(f"Saved layout visualization to: {vis_path}")
                    
                    # 2d. Process Azure text paragraphs
                    azure_elements = process_azure_paragraphs(
                        paragraphs = azure_result.get('paragraphs', []),
                        pdf_name = pdf_path.name,
                        page_info = azure_page_info,
                        page_num = page_num,
                        ignor_roles = self.ignor_roles,
                        min_length = self.min_length
                    )
                    print(f"Processed {len(azure_elements)} Azure text elements")
                    elements.extend(azure_elements)
                    
                    # 2e. Process layout elements with Nougat
                    layout_records = self._process_layout_elements(
                        layout_elements,
                        page_img,
                        pdf_path.name,
                        page_num
                    )
                    print(f"Processed {len(layout_records)} layout elements")
                    elements.extend(layout_records)
                    
                except Exception as e:
                    print(f"Error processing page {page_num}: {str(e)}")
                    continue  # Continue with next page
            
            # Step 3: Create and process DataFrame
            print("\nStep 3: Element Processing")
            df = self._create_dataframe(elements)
            print(f"Initial DataFrame rows: {len(df)}")
            
            # 3a. Normalize bounding boxes
            print("Normalizing bounding boxes...")
            df = bbox_scaler.normalize_bounding_boxes(df)
            
            # 3b. Filter overlapping elements
            print("Filtering overlapping elements...")
            df = filter_overlapping_elements(df, self.overlap_threshold)
            print(f"Rows after overlap filtering: {len(df)}")
            
            # 3c. Filter margin elements
            print("Filtering margin elements...")
            margin_detector = MarginDetector(
                density_bins=500,
                min_column_gap=50,
                peak_prominence=0.9
            )

            # Get margins and column layout
            margin_sizes, column_layout = margin_detector.detect_margins(df)

            # Filter elements
            df = margin_detector.filter_margin_elements(df, margin_sizes, column_layout)
            print(f"Rows after margin filtering: {len(df)}")
            
            # Step 4: Generate outputs
            print("\nStep 4: Generating Outputs")
            
            # 4a. Create markdown
            print("Generating markdown...")
            markdown_text = self._create_markdown_from_df(df)
            
            # 4b. Create visualizations
            print("Creating visualizations...")
            visualizer = BoundingBoxVisualizer()
            visualization_paths = visualizer.create_overlay_visualization(
                pdf_path,
                df,
                self.output_dir
            )
            print(f"Created {len(visualization_paths)} page visualizations")
            
            return markdown_text, df, visualization_paths
            
        except Exception as e:
            print(f"\nError processing document: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _create_markdown_from_df(self, df: pd.DataFrame) -> str:
        """
        Create markdown text directly from the DataFrame.
        
        Args:
            df: DataFrame containing document elements
            
        Returns:
            str: Markdown formatted text
        """
        markdown = []
        current_page = 0
        
        # Process elements in order by page and vertical position
        for _, row in df.iterrows():
            if row['page'] != current_page:
                current_page = row['page']
                markdown.append(f"\n## Page {current_page}\n")
            
            if row['type'] == 'text':
                # Add text content
                if row['text']:
                    markdown.append(row['text'])
            elif row['type'] == 'image':
                # Add image reference if path exists
                if row['image_path']:
                    try:
                        rel_path = Path(row['image_path']).relative_to(self.output_dir)
                        markdown.append(f"\n![]({rel_path})\n")
                    except ValueError:
                        markdown.append(f"\n![](/{row['image_path']})\n")
            elif row['type'] in ['table', 'formula']:
                # Add both image and extracted text for tables and formulas
                if row['image_path']:
                    try:
                        rel_path = Path(row['image_path']).relative_to(self.output_dir)
                        markdown.append(f"\n![]({rel_path})\n")
                    except ValueError:
                        markdown.append(f"\n![](/{row['image_path']})\n")
                    
                    # Add extracted text below if available
                    if row['text']:
                        markdown.append(f"\n```\n{row['text']}\n```\n")
            elif 'caption' in row['type']:
                # Add caption text
                if row['text']:
                    markdown.append(f"*{row['text']}*\n")
                    
        return "\n".join(markdown)

    def _process_layout_elements(self,
                                elements: List[DocumentElement],
                                page_img: Image.Image,
                                pdf_name: str,
                                page_num: int) -> List[DocumentElementRecord]:
        """Process elements detected by LayoutDetectionService with Nougat text extraction."""
        records = []
        
        for elem in elements:
            # Skip text elements as they're handled by Azure
            if elem.label.lower() == 'text':
                continue
                
            # Determine element type and extraction method
            extraction_method = 'default'
            if elem.label.lower() in ['figure', 'image', 'graphic', 'photo', 'diagram', 'picture']:
                elem_type = DocumentElementType.IMAGE
            elif elem.label.lower() == 'table':
                elem_type = DocumentElementType.TABLE
                extraction_method = 'nougat'
            elif elem.label.lower() == 'formula':
                elem_type = DocumentElementType.FORMULA
                extraction_method = 'nougat'
            elif elem.label.lower() == 'caption':
                elem_type = self._determine_caption_type(elem, elements)
                extraction_method = 'nougat'
            else:
                continue
                
            # Save element image
            img_path = self._save_element_image(page_img, elem, page_num, self.top_margin, self.bottom_margin)
            
            # Extract text using appropriate method
            if extraction_method == 'nougat' and elem_type != DocumentElementType.IMAGE:
                extracted_text = self._extract_text_with_nougat(elem, page_num, img_path, elem_type)
            else:
                extracted_text = None
            
            # Create record with extracted or original text
            records.append(DocumentElementRecord(
                pdf_file=pdf_name,
                page=page_num,
                bounding_box=BoundingBox.from_layout_box(page_num, elem.box),
                element_type=elem_type,
                text=extracted_text if extracted_text else elem.text,
                image_path=str(img_path).split('output_dir/')[-1],
                confidence=elem.confidence,
                metadata={
                    'extraction_method': extraction_method,
                    'extraction_success': bool(extracted_text),
                    'source': 'layout_detector'
                }
            ))
            
        return records

    def _extract_text_with_nougat(self, elem, page_num, img_path, elem_type):
        extracted_text = None
        try:
            # Try multiple extraction attempts with error handling
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Process with Nougat
                    extracted_text = self.nougat_service.get_text_from_nougat(img_path)
                    if extracted_text:
                        break
                except AttributeError as ae:
                    # Handle specific model attribute errors
                    if 'pos_drop' in str(ae):
                        print(f"Warning: Known model attribute issue encountered (attempt {attempt + 1}/{max_attempts})")
                        continue
                    else:
                        raise ae
                except Exception as e:
                    if attempt < max_attempts - 1:
                        print(f"Extraction attempt {attempt + 1} failed, retrying...")
                        continue
                    else:
                        print(f"Error extracting text with Nougat for {elem.label} on page {page_num}: {str(e)}")
                        break
            
            if not extracted_text:
                # If Nougat extraction failed, try Azure Document Intelligence fallback for tables
                if elem_type == DocumentElementType.TABLE:
                    try:
                        # Use Azure Document Intelligence for OCR fallback
                        azure_result = analyze_page_with_azure(
                            self.azure_client, 
                            img_path, 
                            page_num, 
                            self.output_dir
                        )
                        extracted_text = azure_result["content"]
                    except Exception as e:
                        print(f"Azure OCR fallback failed for table on page {page_num}: {str(e)}")
                else:
                    print(f"Warning: No text extracted from {elem.label} on page {page_num}")
        except Exception as e:
            print(f"Error processing {elem.label} image on page {page_num}: {str(e)}")
        
        return extracted_text

    def _determine_caption_type(self, 
                              caption_elem: DocumentElement,
                              elements: List[DocumentElement]) -> DocumentElementType:
        """Determine caption type based on proximity to other elements."""
        caption_center = (caption_elem.box[1] + caption_elem.box[3]) / 2
        min_distance = float('inf')
        closest_type = DocumentElementType.IMAGE_CAPTION
        
        for elem in elements:
            if elem.label.lower() in ['figure', 'table', 'formula']:
                elem_center = (elem.box[1] + elem.box[3]) / 2
                distance = abs(elem_center - caption_center)
                
                if distance < min_distance:
                    min_distance = distance
                    if elem.label.lower() == 'table':
                        closest_type = DocumentElementType.TABLE_CAPTION
                    elif elem.label.lower() == 'formula':
                        closest_type = DocumentElementType.FORMULA_CAPTION
                    else:
                        closest_type = DocumentElementType.IMAGE_CAPTION
                        
        return closest_type

    def _save_element_image(self, 
                            page_img: Image.Image, 
                            element: DocumentElement,
                            page_num: int,
                            top_margin: int = 0,
                            bottom_margin: int = 0
                          ) -> str:
        """
        Save an element as an image and return the path.

        The `top_margin` and `bottom_margin` parameters specify the number of pixels to 
        expand the bounding box of the detected element before cropping the image. These margins are used to 
        include surrounding context in the cropped image. The unit is pixels.

        Typical pixel dimensions for a standard document page (8.5 x 11 inches) at 300 DPI are approximately 2550 x 3300 pixels.
        The exact number of pixels will vary depending on the DPI and the dimensions of the page.

        Args:
            page_img (Image.Image): The PIL Image object representing the page.
            element (DocumentElement): The document element to save as an image.
            page_num (int): The page number of the element.
            top_margin (int, optional): The margin in pixels to add to the top of the bounding box. Defaults to 0.
            bottom_margin (int, optional): The margin in pixels to add to the bottom of the bounding box. Defaults to 0.

        Returns:
            str: The path to the saved image.
        """
        # Create directory for element type
        element_dir = self.output_dir / 'elements' / element.label.lower()
        # element_dir = f'elements/{element.label.lower()}'
        # os.makedirs(element_dir, exist_ok=True)
        element_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate the boundaries of the image
        width, height = page_img.size
        
        # Crop and save image with margins
        # Ensure the box coordinates are within the image boundaries
        top = max(0, element.box[1] - top_margin)
        bottom = min(height, element.box[3] + bottom_margin)
        box = (
            element.box[0],
            top,
            element.box[2],
            bottom
        )
        
        element_img = page_img.crop(box)
        
        # output_path = f'{element_dir}/page_{page_num}_{element.label}_{id(element)}.png'
        output_path = element_dir / f'page_{page_num}_{element.label}_{id(element)}.png'
        element_img.save(output_path)
        
        return str(output_path)

    def _pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """
        Convert PDF pages to images optimized for processing.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of page images
        """
        doc = pymupdf.open(pdf_path)
        images = []
        
        for page in doc:
            # Use a higher DPI for better text recognition (300 DPI)
            zoom = 3.0  # 72 dpi * 3 = 216 dpi
            matrix = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Convert to PIL Image in RGB mode
            img = Image.frombytes(
                'RGB',
                [pix.width, pix.height],
                pix.samples
            )
            
            images.append(img)
            
        doc.close()
        return images

    def _create_dataframe(self, elements: List[DocumentElementRecord]) -> pd.DataFrame:
        """Create DataFrame from document elements."""
        records = []
        for elem in elements:
            # Get page dimensions from metadata if available
            page_width = elem.metadata.get('page_width') if elem.metadata else None
            page_height = elem.metadata.get('page_height') if elem.metadata else None
            page_unit = elem.metadata.get('page_unit') if elem.metadata else None
            azure_order_id = elem.metadata.get('azure_order_id') if elem.metadata else float('inf')
            
            # Create the bounding box string
            box = f"({elem.bounding_box.x1:.2f}, {elem.bounding_box.y1:.2f}, " \
                f"{elem.bounding_box.x2:.2f}, {elem.bounding_box.y2:.2f})"
                
            record = {
                'pdf_file': elem.pdf_file,
                'page': elem.page,
                'bounding_box': box,
                'type': elem.element_type.value,
                'text': elem.text,
                'image_path': elem.image_path,
                'role': elem.role,
                'confidence': elem.confidence,
                'spans': str(elem.spans) if elem.spans else None,
                'source': elem.metadata.get('source') if elem.metadata else None,
                'page_width': page_width,
                'page_height': page_height,
                'page_unit': page_unit,
                'azure_order_id': azure_order_id  # Add azure_order_id to DataFrame
            }
            records.append(record)
            
        return pd.DataFrame(records)

