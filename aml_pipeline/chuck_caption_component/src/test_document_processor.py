import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import pandas as pd
import json
import os
import sys
from pathlib import Path
import uuid
from document_processor import process_row_worker

class TestProcessRowWorkerAdditional(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock data for testing
        self.api_key = "test_api_key"
        self.endpoint = "https://test-endpoint.openai.azure.com"
        self.vision_deployment_name = "test-deployment"
        self.max_chunk_length = 4000
        self.input_folder = "/test/input/folder"
        self.summary_map = {"test-pdf": "This is a test summary."}
        
        # Create a mock row with empty data
        self.empty_row = pd.Series({
            'text': '',
            'pdf_file': 'test-pdf.pdf',
            'image_path': None,
            'page': 1,
            'role': 'text',
            'type': 'paragraph',
            'confidence': 0.99,
            'source': 'OCR',
            'bounding_box': '0,0,100,100',
            'normalized_box': '[0.0, 0.0, 1.0, 1.0]',
            'url': 'http://example.com'
        })
        
        # Create a mock row with invalid image path
        self.invalid_image_row = pd.Series({
            'text': '',
            'pdf_file': 'test-pdf.pdf',
            'image_path': 'nonexistent_image.jpg',
            'page': 1,
            'role': 'image',
            'type': 'figure',
            'confidence': 0.98,
            'source': 'layout',
            'bounding_box': '0,0,200,200',
            'normalized_box': '[0.0, 0.0, 1.0, 1.0]',
            'url': 'http://example.com'
        })
        
        # Create a mock row with a filtered image class
        self.filtered_image_row = pd.Series({
            'text': '',
            'pdf_file': 'test-pdf.pdf',
            'image_path': 'filtered_image.jpg',
            'page': 1,
            'role': 'image',
            'type': 'figure',
            'confidence': 0.98,
            'source': 'layout',
            'bounding_box': '0,0,200,200',
            'normalized_box': '[0.0, 0.0, 1.0, 1.0]',
            'url': 'http://example.com'
        })
        
        # Create a mock row with long text
        self.long_text_row = pd.Series({
            'text': 'This is a very long text ' * 500,  # Make it long enough to be chunked
            'pdf_file': 'test-pdf.pdf',
            'image_path': None,
            'page': 1,
            'role': 'text',
            'type': 'paragraph',
            'confidence': 0.99,
            'source': 'OCR',
            'bounding_box': '0,0,100,100',
            'normalized_box': '[0.0, 0.0, 1.0, 1.0]',
            'url': 'http://example.com'
        })

    @patch('openai.AzureOpenAI')
    @patch('os.path.exists')
    def test_empty_row_processing(self, mock_exists, mock_azure_openai):
        """Test processing a row with no text and no image returns empty list."""
        # Setup mocks
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client
        mock_exists.return_value = True
        
        # Set up DocumentProcessor mock
        mock_processor = MagicMock()
        mock_processor.chunk_text.return_value = []  # Empty chunks
        
        # Mock the DocumentProcessor class
        with patch('..document_processor.DocumentProcessor', return_value=mock_processor):
            result = process_row_worker(
                0, self.empty_row, self.summary_map, 
                self.input_folder, self.api_key, self.endpoint, 
                self.vision_deployment_name, self.max_chunk_length
            )
            
            # Check result
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 0)  # Should have no chunks
            
            # Verify chunk_text was called
            mock_processor.chunk_text.assert_called_once()

    @patch('openai.AzureOpenAI')
    @patch('os.path.join')
    @patch('os.path.exists')
    def test_invalid_image_path(self, mock_exists, mock_join, mock_azure_openai):
        """Test processing a row with an invalid image path."""
        # Setup mocks
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client
        mock_exists.return_value = False  # Simulate file not existing
        mock_join.return_value = "/test/input/folder/nonexistent_image.jpg"
        
        # Set up DocumentProcessor mock
        mock_processor = MagicMock()
        mock_processor.chunk_text.return_value = []  # No text chunks
        mock_processor.get_image_classification.return_value = None  # Simulate classification failure
        
        # Mock the DocumentProcessor class
        with patch('..document_processor.DocumentProcessor', return_value=mock_processor):
            result = process_row_worker(
                0, self.invalid_image_row, self.summary_map, 
                self.input_folder, self.api_key, self.endpoint, 
                self.vision_deployment_name, self.max_chunk_length
            )
            
            # Should still call get_image_classification
            mock_processor.get_image_classification.assert_called_once()
            
            # Check result - should be empty as both text and image processing failed
            self.assertEqual(result, [])

    @patch('openai.AzureOpenAI')
    @patch('os.path.join')
    @patch('os.path.exists')
    def test_filtered_image_class(self, mock_exists, mock_join, mock_azure_openai):
        """Test processing an image with a filtered class (49, 50, 51)."""
        # Setup mocks
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client
        mock_exists.return_value = True
        mock_join.return_value = "/test/input/folder/filtered_image.jpg"
        
        # Set up DocumentProcessor mock
        mock_processor = MagicMock()
        mock_processor.chunk_text.return_value = []  # No text chunks
        mock_processor.get_image_classification.return_value = {'image_class': 49, 'probability_score': 0.95}  # Filtered class
        
        # Mock the DocumentProcessor class
        with patch('..document_processor.DocumentProcessor', return_value=mock_processor):
            result = process_row_worker(
                0, self.filtered_image_row, self.summary_map, 
                self.input_folder, self.api_key, self.endpoint, 
                self.vision_deployment_name, self.max_chunk_length
            )
            
            # Check that get_image_classification was called but generate_caption was not
            mock_processor.get_image_classification.assert_called_once()
            mock_processor.generate_caption.assert_not_called()
            
            # Should return empty list when image class is filtered
            self.assertEqual(result, [])

    @patch('openai.AzureOpenAI')
    @patch('pathlib.Path')
    @patch('os.path.exists')
    @patch('os.path.getmtime')
    def test_multiple_text_chunks(self, mock_getmtime, mock_exists, mock_path, mock_azure_openai):
        """Test processing text that gets split into multiple chunks."""
        # Setup mocks
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client
        mock_exists.return_value = True
        mock_getmtime.return_value = 12345.0
        
        # Set up DocumentProcessor mock with multiple chunks
        mock_processor = MagicMock()
        mock_processor.chunk_text.return_value = ['Chunk 1', 'Chunk 2', 'Chunk 3']
        mock_processor.get_text_stats.return_value = {'tokens': 2, 'chars': 7, 'lines': 1}
        
        # Mock the DocumentProcessor class
        with patch('..document_processor.DocumentProcessor', return_value=mock_processor):
            result = process_row_worker(
                0, self.long_text_row, self.summary_map, 
                self.input_folder, self.api_key, self.endpoint, 
                self.vision_deployment_name, self.max_chunk_length
            )
            
            # Check result
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 3)  # Should have three chunks
            self.assertEqual(result[0]['content'], 'Chunk 1')
            self.assertEqual(result[1]['content'], 'Chunk 2')
            self.assertEqual(result[2]['content'], 'Chunk 3')
            
            # Verify get_text_stats was called for each chunk
            self.assertEqual(mock_processor.get_text_stats.call_count, 3)

    @patch('openai.AzureOpenAI')
    @patch('os.path.join')
    @patch('os.path.exists')
    def test_image_classification_exception(self, mock_exists, mock_join, mock_azure_openai):
        """Test handling of exceptions during image classification."""
        # Setup mocks
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client
        mock_exists.return_value = True
        mock_join.return_value = "/test/input/folder/images/test-image.jpg"
        
        # Set up DocumentProcessor mock that raises exception
        mock_processor = MagicMock()
        mock_processor.chunk_text.return_value = []  # No text chunks
        mock_processor.get_image_classification.side_effect = Exception("Image classification error")
        
        # Mock the DocumentProcessor class
        with patch('..document_processor.DocumentProcessor', return_value=mock_processor):
            result = process_row_worker(
                0, self.invalid_image_row, self.summary_map, 
                self.input_folder, self.api_key, self.endpoint, 
                self.vision_deployment_name, self.max_chunk_length
            )
            
            # Should still complete without raising the exception
            self.assertEqual(result, [])

    @patch('openai.AzureOpenAI')
    @patch('uuid.uuid4')
    @patch('pathlib.Path')
    @patch('os.path.exists')
    @patch('os.path.getmtime')
    def test_uuid_generation(self, mock_getmtime, mock_exists, mock_path, mock_uuid4, mock_azure_openai):
        """Test that unique UUIDs are generated for chunks and documents."""
        # Setup mocks
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client
        mock_exists.return_value = True
        mock_getmtime.return_value = 12345.0
        
        # Mock UUIDs with predictable values
        mock_uuid4.side_effect = [
            uuid.UUID('12345678-1234-5678-1234-567812345678'),
            uuid.UUID('87654321-8765-4321-8765-432187654321')
        ]
        
        # Set up DocumentProcessor mock
        mock_processor = MagicMock()
        mock_processor.chunk_text.return_value = ['Test chunk']
        mock_processor.get_text_stats.return_value = {'tokens': 2, 'chars': 10, 'lines': 1}
        
        # Mock the DocumentProcessor class
        with patch('..document_processor.DocumentProcessor', return_value=mock_processor):
            result = process_row_worker(
                0, self.text_row, self.summary_map, 
                self.input_folder, self.api_key, self.endpoint, 
                self.vision_deployment_name, self.max_chunk_length
            )
            
            # Check UUIDs in result
            self.assertEqual(result[0]['metadata']['source']['id'], '12345678-1234-5678-1234-567812345678')
            self.assertEqual(result[0]['document_id'], '87654321-8765-4321-8765-432187654321')
            
            # Verify uuid4 was called twice
            mock_uuid4.assert_has_calls([call(), call()])

    @patch('openai.AzureOpenAI')
    def test_null_endpoint(self, mock_azure_openai):
        """Test client initialization with a null endpoint value."""
        # Execute the function with a None endpoint
        process_row_worker(
            0, self.text_row, self.summary_map, 
            self.input_folder, self.api_key, None, 
            self.vision_deployment_name, self.max_chunk_length
        )
        
        # Check if AzureOpenAI was called with correct parameters including str(None)
        mock_azure_openai.assert_called_once_with(
            api_key=self.api_key, 
            api_version="2024-05-01-preview", 
            azure_endpoint="None"
        )

if __name__ == '__main__':
    unittest.main()
        
    self.image_row = pd.Series({
        'text': '',
        'pdf_file': 'test-pdf.pdf',
        'image_path': 'images/test-image.jpg',
        'page': 1,
        'role': 'image',
        'type': 'figure',
        'confidence': 0.98,
        'source': 'layout',
        'bounding_box': '0,0,200,200',
        'normalized_box': '[0.0, 0.0, 1.0, 1.0]',
        'url': 'http://example.com'
    })

    @patch('openai.AzureOpenAI')
    def test_client_initialization_parameters(self, mock_azure_openai):
        """Test that AzureOpenAI client is initialized with correct parameters."""
        # Execute the function
        process_row_worker(
            0, self.text_row, self.summary_map, 
            self.input_folder, self.api_key, self.endpoint, 
            self.vision_deployment_name, self.max_chunk_length
        )
        
        # Check if AzureOpenAI was called with correct parameters
        mock_azure_openai.assert_called_once_with(
            api_key=self.api_key, 
            api_version="2024-05-01-preview", 
            azure_endpoint=str(self.endpoint)
        )

    @patch('openai.AzureOpenAI')
    @patch('pathlib.Path')
    @patch('os.path.exists')
    @patch('os.path.getmtime')
    def test_process_row_with_text(self, mock_getmtime, mock_exists, mock_path, mock_azure_openai):
        """Test processing a row with text content."""
        # Setup mocks
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client
        mock_exists.return_value = True
        mock_getmtime.return_value = 12345.0
        
        # Set up DocumentProcessor mock
        mock_processor = MagicMock()
        mock_processor.chunk_text.return_value = ['Chunked text']
        mock_processor.get_text_stats.return_value = {'tokens': 3, 'chars': 12, 'lines': 1}
        
        # Mock the DocumentProcessor class
        with patch('..document_processor.DocumentProcessor', return_value=mock_processor):
            result = process_row_worker(
                0, self.text_row, self.summary_map, 
                self.input_folder, self.api_key, self.endpoint, 
                self.vision_deployment_name, self.max_chunk_length
            )
            
            # Check result
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)  # Should have one chunk
            self.assertEqual(result[0]['content'], 'Chunked text')
            
            # Verify AzureOpenAI client was initialized
            mock_azure_openai.assert_called_once()

    @patch('openai.AzureOpenAI')
    def test_exception_handling(self, mock_azure_openai):
        """Test that exceptions during processing are caught and handled."""
        # Make the AzureOpenAI constructor raise an exception
        mock_azure_openai.side_effect = Exception("Test exception")
        
        # Execute the function - it should not raise the exception
        result = process_row_worker(
            0, self.text_row, self.summary_map, 
            self.input_folder, self.api_key, self.endpoint, 
            self.vision_deployment_name, self.max_chunk_length
        )
        
        # It should return an empty list when exception occurs
        self.assertEqual(result, [])

    @patch('openai.AzureOpenAI')
    @patch('os.path.join')
    @patch('os.path.exists')
    def test_process_row_with_image(self, mock_exists, mock_join, mock_azure_openai):
        """Test processing a row with an image."""
        # Setup mocks
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client
        mock_exists.return_value = True
        mock_join.return_value = "/test/input/folder/images/test-image.jpg"
        
        # Set up DocumentProcessor mock with image classification
        mock_processor = MagicMock()
        mock_processor.chunk_text.return_value = []  # No text chunks
        mock_processor.get_image_classification.return_value = {'image_class': 1, 'probability_score': 0.95}
        mock_processor.generate_caption.return_value = "This is a caption for the image."
        mock_processor.get_text_stats.return_value = {'tokens': 8, 'chars': 36, 'lines': 1}
        
        # Mock the DocumentProcessor class
        with patch('..document_processor.DocumentProcessor', return_value=mock_processor):
            result = process_row_worker(
                0, self.image_row, self.summary_map, 
                self.input_folder, self.api_key, self.endpoint, 
                self.vision_deployment_name, self.max_chunk_length
            )
            
            # Check that image processing was done
            mock_processor.get_image_classification.assert_called_once()
            
            # Check result
            self.assertIsInstance(result, list)
            if len(result) > 0:  # Will have content if image caption was generated
                self.assertEqual(result[0]['content'], "This is a caption for the image.")

    @patch('openai.AzureOpenAI')
    def test_api_version_is_correct(self, mock_azure_openai):
        """Test that AzureOpenAI client is initialized with correct API version."""
        # Execute the function
        process_row_worker(
            0, self.text_row, self.summary_map, 
            self.input_folder, self.api_key, self.endpoint, 
            self.vision_deployment_name, self.max_chunk_length
        )
        
        # Get the parameters used to initialize the client
        args, kwargs = mock_azure_openai.call_args
        
        # Check if the API version is '2024-05-01-preview' as specified in the code
        self.assertEqual(kwargs['api_version'], "2024-05-01-preview")

if __name__ == '__main__':
    unittest.main()