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
from typing import Dict, List
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
        max_chunk_length: int = 4000
    ):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.vision_client = openai_client
        self.vision_deployment_name = vision_deployment_name
        self.embedding_client = embedding_client
        self.embd_deployment_name = embd_deployment_name
        self.index_name = index_name
        self.max_chunk_length = max_chunk_length
        
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
            SimpleField(name="type", type=SearchFieldDataType.String),
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
    
    def encode_image(self, image_path: str) -> str:
        """Encode image from blob storage as base64."""
        try:
            with Image.open(image_path) as img:
                image_data = img.tobytes()
            return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            return None

    def generate_caption(self, image_path: str) -> str:
        """Generate image caption using GPT-4 Vision."""
        try:
            base64_image = self.encode_image(image_path)
            if not base64_image:
                return ""

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
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {str(e)}")
            return ""

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

    def process(self) -> Dict:
        """Main processing function."""
        try:
            # Create search index
            self.create_search_index()
            
            # Read input CSV
            input_csv = list(self.input_folder.glob("*.csv"))[0]
            df = pd.read_csv(input_csv)
            
            # Process and upload documents
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
                        embedding = self.get_embedding(chunk)
                        if embedding:
                            doc = {
                                "id": str(uuid.uuid4()),
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
            
            # Save processing stats
            stats["timestamp"] = datetime.now().isoformat()
            self.output_folder.mkdir(parents=True, exist_ok=True)
            
            with open(self.output_folder / "processing_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Processing completed. Stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            raise