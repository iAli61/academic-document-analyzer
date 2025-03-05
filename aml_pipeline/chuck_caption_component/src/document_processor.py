#!/usr/bin/env python
import tiktoken
import json
import logging
import pandas as pd
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import io
import os
import base64
from tiktoken.core import Encoding
from openai import AzureOpenAI
from .prompt import *
from jinja2 import Template
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ensure logs propagate to Azure ML by adding a StreamHandler to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

# Define a simple log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Avoid adding multiple handlers if the logger already has handlers
if not logger.handlers:
    logger.addHandler(handler)

logger.propagate = True

class_names = {
    1: "Data table",
    2: "Statistical table",
    3: "Line graph",
    4: "Bar chart",
    5: "Scatter plot",
    6: "Box plot",
    7: "Histogram",
    8: "Pie chart",
    9: "Heat map",
    10: "Network graph",                
    11: "Time series plot",
    12: "Light microscopy",
    13: "Electron microscopy",
    14: "Fluorescence microscopy",
    15: "Confocal microscopy",
    16: "Mass spectroscopy",
    17: "NMR spectroscopy",
    18: "IR spectroscopy",
    19: "UV-vis spectroscopy",
    20: "X-ray spectroscopy",
    21: "X-ray (medical)",
    22: "CT scan",
    23: "MRI scan",
    24: "Ultrasound",
    25: "PET scan",
    26: "Histology slide",
    27: "Western blot",
    28: "Gel electrophoresis",
    29: "Chemical structure",
    30: "Molecular diagram",
    31: "Anatomical illustration",
    32: "Flowchart",
    33: "Schematic diagram",
    34: "Circuit diagram",
    35: "Mechanical diagram",
    36: "Process flow diagram",
    37: "Geographic map",
    38: "GIS visualization",    
    39: "Satellite image",
    40: "Equation/Mathematical expression",
    41: "Geometric figure",
    42: "3D rendering",     
    43: "Computer simulation",
    44: "Field photograph",
    45: "Sample photograph",
    46: "Experimental setup",
    47: "Equipment photograph",
    48: "Screenshot",
    49: "Logo/Institutional insignia",
    50: "Infographic",
    51: "other"
}

def load_custom_tiktoken(tiktoken_path):
    """
    Load a custom tiktoken vocabulary file with base64 encoded format
    
    Args:
        tiktoken_path: Path to your .tiktoken file
    Returns:
        CustomEncoding object
    """
    # Load the .tiktoken file and parse base64 entries
    vocab = {}
    with open(tiktoken_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                token_base64, rank_str = line.strip().split()
                # Decode base64 to bytes
                token_bytes = base64.b64decode(token_base64)
                rank = int(rank_str)
                vocab[token_bytes] = rank

    # Use the same pattern as cl100k_base
    pat_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]??[^\r\n\p{L}\p{N}]'|\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # pat_str=r"[a-zA-Z0-9]+|[^a-zA-Z0-9\s]+"
    
    # Create the custom encoding
    encoding = Encoding(
        name="cl100k_custom",
        pat_str=pat_str,
        mergeable_ranks=vocab,
        special_tokens={}  # Add any special tokens if needed
    )
    
    return encoding

current_file_path = Path(__file__).resolve().parent
tiktoken_file_path = current_file_path / "tiktoken_files/cl100k_base.tiktoken"
encoding = load_custom_tiktoken(tiktoken_file_path)

# Test the encoding
test_text = "Hello world!"
tokens = encoding.encode(test_text)
decoded = encoding.decode(tokens)

print(f"Encoded tokens: {tokens}")
print(f"Decoded text: {decoded}")

class DocumentProcessor:
    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        openai_client: AzureOpenAI,
        vision_deployment_name: str,
        max_chunk_length: int = 4000,
        max_image_size: int = 20971520  # 20MB limit for GPT-4V
    ):
        self.input_folder = Path(input_folder)
        logger.info(f"Input folder: {self.input_folder}")
        self.output_folder = Path(output_folder)
        logger.info(f"Output folder: {self.output_folder}")
        self.vision_client = openai_client
        self.vision_deployment_name = vision_deployment_name
        self.summary_client = openai_client
        self.summary_deployment_name = vision_deployment_name
        self.max_chunk_length = max_chunk_length
        self.max_image_size = max_image_size
        
        # Initialize text splitter with fallback method
        self._initialize_text_splitter()

    def validate_image(self, image_path: str) -> bool:
        """Validate if the image is suitable for processing."""
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image file does not exist: {image_path}")
                
                return False

            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size > self.max_image_size:
                logger.warning(f"Image too large ({file_size} bytes): {image_path}")
                return False

            # Verify image can be opened and is valid
            with Image.open(image_path) as img:
                img.verify()
                
            # Check if image has valid dimensions
            with Image.open(image_path) as img:
                width, height = img.size
                if width == 0 or height == 0:
                    logger.warning(f"Invalid image dimensions: {image_path}")
                    return False
                    
                if width > 32768 or height > 32768:
                    logger.warning(f"Image dimensions too large: {width}x{height}")
                    return False

            return True

        except Exception as e:
            logger.warning(f"Image validation failed for {image_path}: {str(e)}")
            return False

    def preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Validate if the image is suitable for processing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                max_dimension = 2048
                if img.width > max_dimension or img.height > max_dimension:
                    ratio = min(max_dimension / img.width, max_dimension / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                img_byte_arr.seek(0)
                
                return Image.open(img_byte_arr)

        except Exception as e:
            logger.error(f"Image preprocessing failed for {image_path}: {str(e)}")
            return None

    def encode_image(self, image_path: str) -> Optional[str]:
        """Encode image as base64 with proper validation and preprocessing."""
        try:
            if not self.validate_image(image_path):
                return None

            processed_img = self.preprocess_image(image_path)
            if processed_img is None:
                return None

            img_byte_arr = io.BytesIO()
            processed_img.save(img_byte_arr, format='JPEG', quality=85)
            img_byte_arr.seek(0)
            base64_encoded = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            try:
                base64.b64decode(base64_encoded)
                return base64_encoded
            except Exception as e:
                logger.error(f"Base64 validation failed for {image_path}: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Image encoding failed for {image_path}: {str(e)}")
            return None

    def get_image_classification(self, image_path):
        try:
            base64_image = self.encode_image(image_path)
            if not base64_image:
                logger.info(f"Skipping caption generation for invalid image: {image_path}")
                return ""

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.vision_client.chat.completions.create(
                        model = self.vision_deployment_name,
                        response_format={ "type": "json_object" },
                        messages=[
                            {
                                "role": "system",
                                "content": IMAGE_CLASSIFICATION_SYSTEM_PROMPT
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", 
                                     "text": IMAGE_CLASSIFICATION_USER_PROMPT
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0.0,
                        max_tokens=100
                    )
                    return json.loads(response.choices[0].message.content)

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.info(f"Image classification attempt {attempt + 1} failed: {str(e)}")
                        continue
                    else:
                        logger.info(f"All image classification attempts failed for {image_path}")
                        return None

        except Exception as e:
            logger.info(f"Image classification failed: {str(e)}")
            return None


    def get_summary(self, raw_text):
        try:
            template = Template(DOCUMENT_SUMMARIZATION_USER_PROMPT)
            user_prompt = template.render(DOCUMENT_TEXT=raw_text)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.summary_client.chat.completions.create(
                        model = self.summary_deployment_name,
                        messages=[
                            {
                                "role": "system",
                                "content": DOCUMENT_SUMMARIZATION_SYSTEM_PROMPT
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", 
                                     "text": user_prompt
                                    }
                                ]
                            }
                        ],
                        temperature=0.0,
                        max_tokens=4000
                    )
                    return response.choices[0].message.content

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.info(f"Summary generation attempt {attempt + 1} failed: {str(e)}")
                        continue
                    else:
                        logger.info(f"All summary generation attempts failed for {raw_text[:100]}")
                        return ""

        except Exception as e:
            logger.info(f"Summary generation failed: {str(e)}")
            return ""
        
    def generate_summary_text(self, pdf_group): # Assuming this is your function
        """
        Generates a summary text for a group of rows belonging to the same PDF.
        (Replace this with your actual summary generation logic)
        """
        dir_path = pdf_group['dir_path'].iloc[0] # get pdf file name from the group
        md_file_path = f"{self.input_folder}/{dir_path}/{dir_path}_analysis.md"
        logger.info(md_file_path)

        # Read the contents of the file  
        with open(md_file_path, 'r') as md_file:
            
            md_content = md_file.read()
            logger.info(f"md text: {md_content[:100]}...")

        summary_text = self.get_summary(md_content)
        logger.info(f"summary: {summary_text[:100]}...")
        return summary_text

    def generate_caption(self, image_path, figure_title, document_summary) -> str:
        """Generate image caption using GPT-4 Vision."""
        try:
            base64_image = self.encode_image(image_path)
            if not base64_image:
                logger.info(f"Skipping caption generation for invalid image: {image_path}")
                return ""

            template = Template(IMAGE_CAPTIONING_USER_PROMPT)
            user_prompt = template.render(IMAGE_TITLE=figure_title, DOCUMENT_CONTEXT=document_summary)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.vision_client.chat.completions.create(
                        model = self.vision_deployment_name,
                        messages=[
                            {
                                "role": "system",
                                "content": IMAGE_CAPTIONING_SYSTEM_PROMPT
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", 
                                     "text": user_prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0.0,
                        max_tokens=2000
                    )
                    return response.choices[0].message.content

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.info(f"Caption generation attempt {attempt + 1} failed: {str(e)}")
                        continue
                    else:
                        logger.info(f"All caption generation attempts failed for {image_path}")
                        return ""

        except Exception as e:
            logger.info(f"Caption generation failed for {image_path}: {str(e)}")
            return ""

    def get_text_stats(self, text: str) -> Dict:
        """Calculate text statistics including tokens, characters, and lines."""
        if pd.isna(text) or not text:
            return {"tokens": 0, "chars": 0, "lines": 0}
            
        return {
            "tokens": len(self.tokenizer.encode(text)),
            "chars": len(text),
            "lines": len(text.splitlines())
        }

    def _initialize_text_splitter(self):
        """Initialize text splitting functionality with fallback options."""

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("Successfully initialized tiktoken tokenizer")
            self.split_text = self._split_text_tiktoken
        # if it fails, load custom tiktoken
        except Exception as e:
            self.tokenizer = load_custom_tiktoken("./src/tiktoken_files/cl100k_base.tiktoken")
            logger.info("Successfully initialized custom tiktoken tokenizer")
            self.split_text = self._split_text_tiktoken
        
        if not self.tokenizer:
            logger.warning("Failed to initialize tiktoken tokenizer")
            logger.info("Falling back to basic text splitting")
            self.split_text = self._split_text_basic


    def _split_text_tiktoken(self, text: str) -> List[str]:
        """Split text using tiktoken tokenizer."""
        if pd.isna(text) or len(text) <= self.max_chunk_length:
            return [text] if pd.notna(text) else []

        tokens = self.tokenizer.encode(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for token in tokens:
            token_text = self.tokenizer.decode([token])
            token_length = len(token_text)
            
            if current_length + token_length > self.max_chunk_length:
                if current_chunk:
                    chunks.append(self.tokenizer.decode(current_chunk))
                current_chunk = [token]
                current_length = token_length
            else:
                current_chunk.append(token)
                current_length += token_length

        if current_chunk:
            chunks.append(self.tokenizer.decode(current_chunk))

        return chunks

    def _split_text_basic(self, text: str) -> List[str]:
        """Fallback method for text splitting using basic string operations."""
        if pd.isna(text) or len(text) <= self.max_chunk_length:
            return [text] if pd.notna(text) else []

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) + 1 > self.max_chunk_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence) + 1  # +1 for space

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """Public method for text chunking that uses the initialized splitter."""
        return self.split_text(text)

    def process(self) -> Tuple[Dict, str]:
        """Process documents and return stats and output file path."""
        try:
            csv_files = list(self.input_folder.glob("*.csv"))
            if not csv_files:
                logger.error(f"No CSV files found in input folder: {self.output_folder}")
                return ({
                    "error": "No input CSV files found",
                    "timestamp": datetime.now().isoformat()
                }, "")

            input_csv = csv_files[0]
            logger.info(f"Processing CSV file: {input_csv}")
            df = pd.read_csv(input_csv)

            if df.empty:
                logger.error("Input CSV file is empty")
                return ({
                    "error": "Empty input CSV file",
                    "timestamp": datetime.now().isoformat()
                }, "")

            stats = {
                "processed": 0,
                "errors": 0,
                "chunks": 0,
                "input_file": str(input_csv),
                "total_rows": len(df)
            }

            df['dir_path'] = df['pdf_file'].str.replace('.pdf', '')

            # Generate summaries in parallel using ThreadPoolExecutor
            logger.info("Generating summaries in parallel...")
            pdf_groups = df.groupby('dir_path')
            summary_map = {}
            with ThreadPoolExecutor() as executor:
                future_to_pdf = {executor.submit(self.generate_summary_text, group): name for name, group in pdf_groups}
                for future in as_completed(future_to_pdf):
                    pdf_name = future_to_pdf[future]
                    try:
                        summary_text = future.result()
                        summary_map[pdf_name] = summary_text # Store summary in map
                        logger.info(f"Summary generated for {pdf_name[:50]}...: {summary_text[:50]}...")
                    except Exception as exc:
                        logger.error(f"Summary generation for {pdf_name} failed: {exc}")
                        summary_map[pdf_name] = "" # Store empty string if summary fails

            logger.info("Summaries generated.")

            # Create output directory
            self.output_folder.mkdir(parents=True, exist_ok=True)
            output_file = self.output_folder / "processed_chunks.jsonl"

            processed_rows_count = 0 # Counter for processed rows
            all_chunks_data = [] # List to collect chunk records from all rows

            logger.info("Processing rows in parallel...")
            api_key = self.vision_client.api_key
            endpoint = self.vision_client.base_url

            with multiprocessing.Pool(processes=8) as pool:
                row_results = pool.starmap(
                    process_row_worker,
                    [
                        (index, row, summary_map, self.input_folder, api_key, endpoint, self.vision_deployment_name, self.max_chunk_length)
                        for index, row in df.iterrows()
                    ]
                )

                with open(output_file, 'w', encoding='utf-8') as f:
                    for row_chunk_data in row_results: # Iterate through results from parallel processing
                        for chunk_record in row_chunk_data: # Iterate through chunks returned for each row
                            f.write(json.dumps(chunk_record) + '\n')
                            stats["chunks"] += 1
                        stats["processed"] += 1
                        processed_rows_count += 1
                        if processed_rows_count > 0 and processed_rows_count % 100 == 0:
                            logger.info(f"Processed {processed_rows_count}/{len(df)} rows ({processed_rows_count/len(df)*100:.1f}%)")


            stats["timestamp"] = datetime.now().isoformat()
            stats["completion_status"] = "success"

            # Save stats
            stats_file = self.output_folder / "processing_stats.json"
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)

            logger.info(f"Processing completed. Stats saved to: {stats_file}")
            logger.info(f"Processed chunks saved to: {output_file}")
            logger.info(f"Final stats: {stats}")
            

            return (stats, str(output_file))

        except Exception as e:
            error_stats = {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "completion_status": "failed"
            }
            logger.error(f"Critical error in document processing: {str(e)}")
            return (error_stats, "")

def process_row_worker(index, row, summary_map, input_folder, api_key, endpoint, vision_deployment_name, max_chunk_length):
    try:
        # Initialize variables that might be conditionally assigned
        img_class = None
        probability_score = None

        logger.info(f"azure endpoint: {str(endpoint)}")
        
        # Initialize AzureOpenAI client inside the worker
        vision_client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-05-01-preview",
            azure_endpoint=str(endpoint).replace("/openai/", "/")
        )
      
        has_text = not pd.isna(row['text']) and len(row['text']) > 0
        if index % 1 == 0:  # Log only periodically
            if has_text:
                logger.info(f"Row {index} - Has text: {has_text}, Text length: {len(row['text'])}")
            else:
                logger.info(f"Row {index} - No text found")
        
        # Initialize DocumentProcessor instance locally (minimal setup)
        processor = DocumentProcessor(
            input_folder=input_folder,
            output_folder="temp",  # Temporary placeholder
            openai_client=vision_client,
            vision_deployment_name=vision_deployment_name,
            max_chunk_length=max_chunk_length
        )

        chunks = processor.chunk_text(row['text'])
        logger.info(f"Row {index} - Chunks from text: {len(chunks)}")

        document_summary = summary_map.get(row['pdf_file'].replace('.pdf', ''), '')

        if not chunks and pd.notna(row['image_path']):
            # Log image processing
            if index % 1 == 0:  # Log only periodically
                logger.info(f"Row {index} - Processing image: {row['image_path']}")
                
            # Use string joining instead of Path object for safer handling
            image_path = os.path.join(input_folder, str(row['image_path']))
            
            img_class_result = processor.get_image_classification(image_path)
            logger.info(f"Row {index} - Image classification result: {img_class_result}")
            img_class = img_class_result.get('image_class') if img_class_result else None
            probability_score = img_class_result.get('probability_score') if img_class_result else None
            if img_class is not None and img_class > 48:
                caption = processor.generate_caption(
                    image_path=image_path,
                    figure_title=None,
                    document_summary=document_summary
                )
                chunks = [caption] if caption else []
                
                if index % 500 == 0:  # Log only periodically
                    logger.info(f"Row {index} - Caption generated: {len(caption) if caption else 0} chars")
            else:
                chunks = []
                if index % 500 == 0 and img_class is not None:  # Log only periodically
                    logger.info(f"Row {index} - Image class {img_class} filtered out")

        # Final check on chunks
        if not chunks and index % 500 == 0:  # Log only periodically
            logger.info(f"Row {index} - No chunks generated")

        processed_chunks_data = []
        for chunk in chunks:
            chunk_record = {
                "content": chunk,
                "metadata": {
                    "page_number": int(row['page']),
                    "stats": processor.get_text_stats(chunk),
                    "source": {
                        "filename": row['pdf_file'],
                        "url": str(row.get('url', '')),
                        "mtime": os.path.getmtime(Path(input_folder) / row['pdf_file']) if os.path.exists(Path(input_folder) / row['pdf_file']) else None,
                        "role": str(row['role']),
                        "type": str(row['type']),
                        "image_path": str(row['image_path']) if pd.notna(row['image_path']) else None,
                        "confidence": row['confidence'],
                        "source": str(row['source']),
                        "bounding_box": str(row['bounding_box']),
                        "normalized_box": row['normalized_box'],
                        "image_class_no": img_class if img_class is not None else None,
                        "image_class_prob": probability_score if probability_score is not None else None,
                        "image_class_name": class_names.get(img_class) if img_class is not None else None,
                        "page": row['page'],
                        "summary": document_summary,
                        "id": str(uuid.uuid4())
                    }
                },
                "document_id": str(uuid.uuid4())
            }
            processed_chunks_data.append(chunk_record)

        return processed_chunks_data

    except Exception as e:
        logger.error(f"Error processing row {index}: {str(e)}")
        return []