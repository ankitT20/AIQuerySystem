# LLM interface for generating responses using Google Gemini 

import os
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types # type: ignore

class GeminiLLM:
    """Google Gemini LLM interface"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash"
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using Gemini API"""
        
        # Prepare context
        context_text = "\n\n".join([
            f"Source: {chunk['source']}\nContent: {chunk['text']}"
            for chunk in context_chunks
        ])
        
        sources = list(set([chunk['source'] for chunk in context_chunks]))
        
        # Create prompt
        prompt = f"""You are a helpful AI assistant. Answer the user's question based only on the provided context. 
Always cite the sources from which you draw information. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context provided above."""
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(),
        )
        
        # Generate response using streaming
        response_text = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                response_text += chunk.text
        
        return {
            'response': response_text.strip(),
            'sources': sources,
            'model': self.model,
            'context_used': len(context_chunks)
        }


class LLMManager:
    """Manages LLM provider"""
    
    def __init__(self):
        try:
            self.llm = GeminiLLM()
        except (ImportError, ValueError) as e:
            print(f"Warning: Could not initialize Gemini LLM: {e}")
            print("Please ensure GEMINI_API_KEY is set and google-genai is installed.")
            # For demonstration, we'll create a simple fallback that explains the requirement
            self.llm = None
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using Gemini"""
        if self.llm is None:
            # Provide a helpful response when LLM is not available
            sources = list(set([chunk['source'] for chunk in context_chunks]))
            context_summary = "\n\n".join([f"From {chunk['source']}: {chunk['text'][:100]}..." 
                                          for chunk in context_chunks[:2]])
            
            return {
                'response': f"""Based on the available documents, here's relevant information for "{query}": {context_summary} """,
                'sources': sources,
                'model': 'system-fallback',
                'context_used': len(context_chunks)
            }
        
        return self.llm.generate_response(query, context_chunks)


if __name__ == "__main__":
    # Test the LLM manager
    from document_processor import DocumentProcessor
    from embeddings import VectorStore
    
    # Load and process documents
    processor = DocumentProcessor()
    chunks = processor.process_documents()
    
    # Create vector store
    vector_store = VectorStore()
    vector_store.add_documents(chunks)
    
    # Test query
    query = "What is artificial intelligence?"
    context_chunks = vector_store.similarity_search(query, k=3)
    
    # Test LLM response
    llm_manager = LLMManager()
    response = llm_manager.generate_response(query, context_chunks)
    
    print(f"Query: {query}")
    print(f"Response: {response['response']}")
    print(f"Sources: {response['sources']}")
    print(f"Model: {response['model']}")