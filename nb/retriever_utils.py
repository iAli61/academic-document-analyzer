import ipywidgets as widgets
from IPython.display import display, HTML
from PIL import Image
import pandas as pd
import io
import os

class ImageCaptionViewer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the image viewer widget.
        
        Args:
            df: DataFrame containing image_path and text (caption) columns
        """
        self.df = df
        self.current_index = 0
        
        # Create widgets
        self.image_widget = widgets.Image(
            layout=widgets.Layout(width='800px', height='auto')
        )
        
        self.caption_widget = widgets.HTML(
            layout=widgets.Layout(width='800px')
        )
        
        self.prev_button = widgets.Button(
            description='Previous',
            button_style='info',
            icon='arrow-left'
        )
        
        self.next_button = widgets.Button(
            description='Next',
            button_style='info',
            icon='arrow-right'
        )
        
        self.index_label = widgets.HTML(
            layout=widgets.Layout(width='200px')
        )
        
        # Set up button callbacks
        self.prev_button.on_click(self.previous_image)
        self.next_button.on_click(self.next_image)
        
        # Create layout
        self.button_box = widgets.HBox([
            self.prev_button, 
            self.index_label,
            self.next_button
        ])
        
        self.widget_box = widgets.VBox([
            self.image_widget,
            self.caption_widget,
            self.button_box
        ])
        
        # Display first image
        self.update_display()
        
    def update_display(self):
        """Update the display with current image and caption."""
        if len(self.df) == 0:
            self.caption_widget.value = "<p>No images found</p>"
            return
            
        row = self.df.iloc[self.current_index]
        
        try:
            # Load and display image
            if os.path.exists(row['image_path']):
                with open(row['image_path'], 'rb') as f:
                    self.image_widget.value = f.read()
            else:
                self.image_widget.value = b''
                self.caption_widget.value = f"<p style='color: red'>Image not found: {row['image_path']}</p>"
            
            # Update caption
            caption_html = f"""
            <div style='padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>
                <p style='margin-bottom: 5px'><strong>Caption:</strong> {row['text']}</p>
                <p style='margin-bottom: 5px'><strong>PDF:</strong> {row['pdf_file']}</p>
                <p style='margin-bottom: 5px'><strong>Page:</strong> {row['page']}</p>
                <p style='margin-bottom: 0'><strong>Confidence:</strong> {row['confidence']}</p>
            </div>
            """
            self.caption_widget.value = caption_html
            
            # Update index label
            self.index_label.value = f"<p style='text-align: center; margin: 0;'>{self.current_index + 1} / {len(self.df)}</p>"
            
        except Exception as e:
            self.caption_widget.value = f"<p style='color: red'>Error displaying image: {str(e)}</p>"
    
    def previous_image(self, b):
        """Display previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def next_image(self, b):
        """Display next image."""
        if self.current_index < len(self.df) - 1:
            self.current_index += 1
            self.update_display()
    
    def display(self):
        """Display the widget."""
        display(self.widget_box)

# Example usage in Jupyter notebook:
"""
# First get your DataFrame from Azure Search
retriever = ImageChunkRetriever(
    search_endpoint=search_endpoint,
    search_key=search_key,
    index_name=index_name
)
df = retriever.get_image_chunks()

# Create and display the viewer
viewer = ImageCaptionViewer(df)
viewer.display()
"""