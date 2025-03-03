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

logger = logging.getLogger(__name__)

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

# encoding = load_custom_tiktoken("./src/tiktoken_files/cl100k_base.tiktoken")

# Test the encoding
# test_text = "Hello world!"
# tokens = encoding.encode(test_text)
# decoded = encoding.decode(tokens)

# print(f"Encoded tokens: {tokens}")
# print(f"Decoded text: {decoded}")

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

    def generate_caption(self, image_path: str) -> str:
        """Generate image caption using GPT-4 Vision."""
        try:
            base64_image = self.encode_image(image_path)
            if not base64_image:
                logger.warning(f"Skipping caption generation for invalid image: {image_path}")
                return ""

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.vision_client.chat.completions.create(
                        model=self.vision_deployment_name,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an AI specialized in generating precise, technical captions for academic and scientific images. Focus on quantitative details, scientific notation, chemical formulas, experimental conditions, and technical specifications when present. For graphs and plots, describe key trends, axes, and significant data points. For experimental setups, include relevant equipment and configurations. Maintain scientific accuracy and use field-specific terminology appropriately."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", 
                                     "text": "Generate a detailed, comprehensive caption for this academic image. Include: 1) The type of visualization (e.g., graph, experimental setup, chemical structure, microscopy image, etc.), 2) Key quantitative information or trends if present, 3) Important experimental conditions or parameters if shown, 4) Any chemical formulas, reactions, or mathematical equations if present. Focus on precision and completeness while maintaining conciseness."
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
                        max_tokens=100
                    )
                    return response.choices[0].message.content

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Caption generation attempt {attempt + 1} failed: {str(e)}")
                        continue
                    else:
                        logger.error(f"All caption generation attempts failed for {image_path}")
                        return ""

        except Exception as e:
            logger.error(f"Caption generation failed for {image_path}: {str(e)}")
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
            
            # Create output directory
            self.output_folder.mkdir(parents=True, exist_ok=True)
            output_file = self.output_folder / "processed_chunks.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for idx, row in df.iterrows():
                    try:
                        chunks = self.chunk_text(row['text'])
                        
                        # If no text, generate caption for image
                        if not chunks and pd.notna(row['image_path']):
                            image_path = str(self.input_folder / row['image_path'])
                            caption = self.generate_caption(image_path)
                            chunks = [caption] if caption else []
                        
                        for chunk in chunks:
                            chunk_record = {
                                "content": chunk,
                                "metadata": {
                                    "page_number": int(row['page']),
                                    "stats": self.get_text_stats(chunk),
                                    "source": {
                                        "filename": row['pdf_file'],
                                        "url": row.get('url', ''),
                                        "mtime": os.path.getmtime(self.input_folder / row['pdf_file']) if os.path.exists(self.input_folder / row['pdf_file']) else None,
                                        "role": row['role'],
                                        "type": row['type'],
                                        "image_path": row['image_path'],
                                        "confidence": row['confidence'],
                                        "source": row['source'],
                                        "bounding_box": row['bounding_box'],
                                        "page": row['page'],
                                        "id": str(uuid.uuid4())
                                    }
                                },
                                "document_id": str(uuid.uuid4())
                            }
                            
                            f.write(json.dumps(chunk_record) + '\n')
                        
                        stats["processed"] += 1
                        stats["chunks"] += 1
                        
                        if idx > 0 and idx % 100 == 0:
                            logger.info(f"Processed {idx}/{len(df)} rows ({idx/len(df)*100:.1f}%)")
                            
                    except Exception as e:
                        logger.error(f"Error processing row {idx}: {str(e)}")
                        stats["errors"] += 1
                        continue
            
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