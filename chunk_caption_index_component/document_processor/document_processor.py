#!/usr/bin/env python
# document_processor.py

import base64
import json
import logging
import pandas as pd
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticField,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticSearch,
)

from openai import AzureOpenAI

logger = logging.getLogger(__name__)

import base64
import json
import logging
import pandas as pd
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import io
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticField,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticSearch,
)

from openai import AzureOpenAI

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        openai_client: AzureOpenAI,
        vision_deployment_name: str,
        embedding_client: AzureOpenAI,
        embd_deployment_name: str,
        search_endpoint: str,
        search_key: str,
        search_api_version: str,
        index_name: str,
        max_chunk_length: int = 4000,
        max_image_size: int = 20971520  # 20MB limit for GPT-4V
    ):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.vision_client = openai_client
        self.vision_deployment_name = vision_deployment_name
        self.embedding_client = embedding_client
        self.embd_deployment_name = embd_deployment_name
        self.index_name = index_name
        self.max_chunk_length = max_chunk_length
        self.max_image_size = max_image_size
        
        # Initialize Azure Search clients
        search_credential = AzureKeyCredential(search_key)
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=search_credential,
            api_version=search_api_version
        )
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=search_credential,
            api_version=search_api_version
        )

    def validate_image(self, image_path: str) -> bool:
        """
        Validate if the image is suitable for processing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if image is valid, False otherwise
        """
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
                    
                # Check if image is ridiculously large
                if width > 32768 or height > 32768:  # Arbitrary large dimension limit
                    logger.warning(f"Image dimensions too large: {width}x{height}")
                    return False

            return True

        except Exception as e:
            logger.warning(f"Image validation failed for {image_path}: {str(e)}")
            return False

    def preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Preprocess image for optimal processing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL.Image or None if processing fails
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB mode if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Resize if too large while maintaining aspect ratio
                max_dimension = 2048  # Maximum dimension for either width or height
                if img.width > max_dimension or img.height > max_dimension:
                    ratio = min(max_dimension / img.width, max_dimension / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                img_byte_arr.seek(0)
                
                return Image.open(img_byte_arr)

        except Exception as e:
            logger.error(f"Image preprocessing failed for {image_path}: {str(e)}")
            return None

    def encode_image(self, image_path: str) -> Optional[str]:
        """
        Encode image as base64 with proper validation and preprocessing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Base64 encoded image or None if processing fails
        """
        try:
            # Validate image
            if not self.validate_image(image_path):
                return None

            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            if processed_img is None:
                return None

            # Convert to base64
            img_byte_arr = io.BytesIO()
            processed_img.save(img_byte_arr, format='JPEG', quality=85)
            img_byte_arr.seek(0)
            base64_encoded = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            # Verify base64 string is valid
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
        """
        Generate image caption using GPT-4 Vision with improved error handling.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Generated caption or empty string if processing fails
        """
        try:
            # Encode image
            base64_image = self.encode_image(image_path)
            if not base64_image:
                logger.warning(f"Skipping caption generation for invalid image: {image_path}")
                return ""

            # Generate caption
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.vision_client.chat.completions.create(
                        model=self.vision_deployment_name,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an AI that generates descriptive captions for images."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Describe this image in one sentence."},
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

    def create_search_index(self):
        """Create or update the search index with vector search capabilities."""
        # Define vector search configuration
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw"
                )
            ]
        )

        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=None,
                keywords_fields=[],
                content_fields=[SemanticField(field_name="text")]
            )
        )

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="pdf_file", type=SearchFieldDataType.String),
            SimpleField(name="page", type=SearchFieldDataType.String),
            SimpleField(name="bounding_box", type=SearchFieldDataType.String),
            SimpleField(name="type", type=SearchFieldDataType.String, filterable=True),  # Added filterable attribute
            SearchableField(name="text", type=SearchFieldDataType.String),
            SimpleField(name="image_path", type=SearchFieldDataType.String),
            SimpleField(name="role", type=SearchFieldDataType.String),
            SimpleField(name="confidence", type=SearchFieldDataType.String),
            SimpleField(name="source", type=SearchFieldDataType.String),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=1536,
                vector_search_profile_name="myHnswProfile"  # Changed from vector_search_profile to vector_search_profile_name
            ),
        ]

        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=SemanticSearch(configurations=[semantic_config])
        )

        self.index_client.create_or_update_index(index)
        logger.info(f"Created/Updated search index: {self.index_name}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Azure OpenAI."""
        try:
            response = self.embedding_client.embeddings.create(
                input=text,
                model=self.embd_deployment_name
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None

    def chunk_text(self, text: str) -> List[str]:
        """Split text into smaller chunks."""
        if pd.isna(text) or len(text) <= self.max_chunk_length:
            return [text] if pd.notna(text) else []

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current_chunk = [], ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.max_chunk_length:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def process(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Main processing function.
        
        Returns:
            Tuple[Dict, pd.DataFrame]: A tuple containing:
                - Dict: Processing statistics
                - DataFrame: Processed chunks with embeddings
        """
        try:
            # Create search index
            self.create_search_index()
            
            # Read input CSV
            input_csv = list(self.input_folder.glob("*.csv"))[0]
            df = pd.read_csv(input_csv)
            
            # Initialize lists for chunk data
            chunk_data = []
            documents = []
            stats = {"processed": 0, "errors": 0, "chunks": 0}
            
            for _, row in df.iterrows():
                try:
                    # Handle text chunks
                    chunks = self.chunk_text(row['text'])
                    
                    # If no text, generate caption for image
                    if not chunks and pd.notna(row['image_path']):
                        caption = self.generate_caption(row['image_path'])
                        chunks = [caption] if caption else []
                    
                    for chunk in chunks:
                        chunk_id = str(uuid.uuid4())
                        embedding = self.get_embedding(chunk)
                        
                        if embedding:
                            # Create document for search index
                            doc = {
                                "id": chunk_id,
                                "pdf_file": str(row['pdf_file']),
                                "page": str(row['page']),
                                "bounding_box": str(row['bounding_box']),
                                "type": str(row['type']),
                                "text": chunk,
                                "image_path": str(row['image_path']),
                                "role": str(row['role']),
                                "confidence": str(row['confidence']),
                                "source": str(row['source']),
                                "vector": embedding
                            }
                            documents.append(doc)
                            
                            # Create record for DataFrame
                            chunk_record = {
                                "chunk_id": chunk_id,
                                "pdf_file": row['pdf_file'],
                                "page": row['page'],
                                "bounding_box": row['bounding_box'],
                                "type": row['type'],
                                "text": chunk,
                                "image_path": row['image_path'],
                                "role": row['role'],
                                "confidence": row['confidence'],
                                "source": row['source'],
                                "embedding": embedding,
                                "is_caption": bool(not row['text'] and pd.notna(row['image_path']))
                            }
                            chunk_data.append(chunk_record)
                            stats["chunks"] += 1
                    
                    stats["processed"] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing row: {str(e)}")
                    stats["errors"] += 1
                    continue
                
                # Upload in batches of 1000
                if len(documents) >= 1000:
                    self.search_client.upload_documents(documents=documents)
                    documents = []
            
            # Upload remaining documents
            if documents:
                self.search_client.upload_documents(documents=documents)
            
            # Create DataFrame from processed chunks
            chunks_df = pd.DataFrame(chunk_data)
            
            # Save processing stats
            stats["timestamp"] = datetime.now().isoformat()
            self.output_folder.mkdir(parents=True, exist_ok=True)
            
            # Save stats and DataFrame
            with open(self.output_folder / "processing_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
                
            chunks_df.to_csv(self.output_folder / "processed_chunks.csv", index=False)
            
            logger.info(f"Processing completed. Stats: {stats}")
            return (stats, chunks_df)  # Explicitly return as tuple
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            return ({}, pd.DataFrame())  # Return empty results on error
