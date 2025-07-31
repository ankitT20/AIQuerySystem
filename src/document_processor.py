# Document processing module for the AI Query System 

import os
import re
from typing import List, Dict, Any
from pathlib import Path

class DocumentProcessor:
    """Handles document loading and text extraction"""
    
    def __init__(self, documents_dir: str = "documents"):
        self.documents_dir = documents_dir
        
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load all documents from the documents directory"""
        documents = []
        
        if not os.path.exists(self.documents_dir):
            raise FileNotFoundError(f"Documents directory '{self.documents_dir}' not found")
            
        for filename in os.listdir(self.documents_dir):
            file_path = os.path.join(self.documents_dir, filename)
            
            if filename.endswith('.txt'):
                content = self._load_text_file(file_path)
                documents.append({
                    'filename': filename,
                    'content': content,
                    'file_path': file_path
                })
                
        return documents
    
    def _load_text_file(self, file_path: str) -> str:
        """Load content from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings in the last 100 characters
                last_period = text.rfind('.', start, end)
                last_question = text.rfind('?', start, end)
                last_exclamation = text.rfind('!', start, end)
                
                best_break = max(last_period, last_question, last_exclamation)
                if best_break > start:
                    end = best_break + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position, ensure we make progress
            if end - overlap <= start:
                start = end  # Ensure we always move forward
            else:
                start = end - overlap
            
        return chunks
    
    def process_documents(self) -> List[Dict[str, Any]]:
        """Process all documents and return chunks with metadata"""
        documents = self.load_documents()
        processed_chunks = []
        
        for doc in documents:
            chunks = self.chunk_text(doc['content'])
            
            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    'text': chunk,
                    'source': doc['filename'],
                    'chunk_id': i,
                    'file_path': doc['file_path']
                })
                
        return processed_chunks


if __name__ == "__main__":
    # Test the document processor
    processor = DocumentProcessor()
    try:
        docs = processor.process_documents()
        print(f"Processed {len(docs)} document chunks")
        
        for i, doc in enumerate(docs[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"Source: {doc['source']}")
            print(f"Text: {doc['text'][:100]}...")
            
    except Exception as e:
        print(f"Error: {e}")