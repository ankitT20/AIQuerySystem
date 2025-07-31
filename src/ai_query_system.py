# Main AI Query System that coordinates document processing, vector search, and LLM responses 

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from document_processor import DocumentProcessor
from embeddings import VectorStore
from llm_interface import LLMManager
from role_filter import RoleFilter, FeedbackAnalyzer


class AIQuerySystem:
    """Main AI Query System class"""
    
    def __init__(self, documents_dir: str = "documents", vector_store_path: str = "vector_store.pkl"):
        self.documents_dir = documents_dir
        self.vector_store_path = vector_store_path
        self.document_processor = DocumentProcessor(documents_dir)
        self.vector_store = VectorStore()
        self.llm_manager = LLMManager()
        self.role_filter = RoleFilter()
        self.feedback_analyzer = FeedbackAnalyzer()
        self.initialized = False
        
    def initialize(self, force_rebuild: bool = False):
        """Initialize the system by processing documents and building vector store"""
        print("Initializing AI Query System...")
        
        # Check if vector store exists and load it
        if os.path.exists(self.vector_store_path) and not force_rebuild:
            print("Loading existing vector store...")
            try:
                self.vector_store.load(self.vector_store_path)
                self.initialized = True
                print("Vector store loaded successfully!")
                return
            except Exception as e:
                print(f"Error loading vector store: {e}")
                print("Rebuilding vector store...")
        
        # Process documents and build vector store
        print("Processing documents...")
        try:
            chunks = self.document_processor.process_documents()
            print(f"Processed {len(chunks)} document chunks")
            
            print("Building vector store...")
            self.vector_store.add_documents(chunks)
            
            # Save vector store
            print("Saving vector store...")
            self.vector_store.save(self.vector_store_path)
            
            self.initialized = True
            print("AI Query System initialized successfully!")
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise
    
    def query(self, question: str, top_k: int = 3, user_role: str = 'public') -> Dict[str, Any]:
        """Process a query and return response with sources"""
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        print(f"Processing query: {question} (User role: {user_role})")
        
        # Search for relevant documents
        print("Searching for relevant documents...")
        context_chunks = self.vector_store.similarity_search(question, k=top_k)
        
        # Apply role-based filtering
        context_chunks = self.role_filter.filter_documents(context_chunks, user_role)
        
        if not context_chunks:
            return {
                'query': question,
                'response': "I couldn't find any relevant information in the documents accessible to your role.",
                'sources': [],
                'timestamp': datetime.now().isoformat(),
                'context_chunks': 0,
                'user_role': user_role
            }
        
        print(f"Found {len(context_chunks)} relevant chunks (after role filtering)")
        
        # Generate response using LLM
        print("Generating response...")
        llm_response = self.llm_manager.generate_response(question, context_chunks)
        
        # Apply response filtering
        filtered_response = self.role_filter.filter_response(llm_response['response'], user_role)
        
        # Compile final response
        result = {
            'query': question,
            'response': filtered_response,
            'sources': llm_response['sources'],
            'model_used': llm_response['model'],
            'context_chunks': llm_response['context_used'],
            'timestamp': datetime.now().isoformat(),
            'similarity_scores': [chunk.get('similarity', 0) for chunk in context_chunks],
            'user_role': user_role,
            'filtered': filtered_response != llm_response['response']
        }
        
        return result
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system status"""
        return {
            'initialized': self.initialized,
            'documents_dir': self.documents_dir,
            'vector_store_path': self.vector_store_path,
            'vector_store_exists': os.path.exists(self.vector_store_path),
            'num_vectors': len(self.vector_store.vectors) if self.initialized else 0,
            'num_documents': len(self.vector_store.metadata) if self.initialized else 0
        }
    
    def list_documents(self) -> List[str]:
        """List available documents"""
        if not os.path.exists(self.documents_dir):
            return []
        
        return [f for f in os.listdir(self.documents_dir) if f.endswith('.txt')]
    
    def add_feedback(self, query: str, response: str, helpful: bool, comments: str = ""):
        """Add feedback for a query response (for future improvements)"""
        feedback = {
            'query': query,
            'response': response,
            'helpful': helpful,
            'comments': comments,
            'timestamp': datetime.now().isoformat()
        }
        
        feedback_file = "feedback.jsonl"
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback) + '\n')
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        return self.feedback_analyzer.get_feedback_stats()


def main():
    """Main function for command-line testing"""
    import sys
    
    # Initialize system
    query_system = AIQuerySystem()
    
    try:
        query_system.initialize()
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        return
    
    # Interactive mode if no arguments provided
    if len(sys.argv) == 1:
        print("\n" + "="*50)
        print("AI Query System - Interactive Mode")
        print("="*50)
        print("Type 'quit' to exit, 'info' for system info, 'docs' to list documents")
        
        while True:
            try:
                query = input("\nEnter your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'info':
                    info = query_system.get_system_info()
                    print("\nSystem Information:")
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                    continue
                elif query.lower() == 'docs':
                    docs = query_system.list_documents()
                    print(f"\nAvailable documents ({len(docs)}):")
                    for doc in docs:
                        print(f"  - {doc}")
                    continue
                
                if not query:
                    continue
                
                # Process query
                result = query_system.query(query)
                
                print(f"\nResponse:")
                print(f"{result['response']}")
                print(f"\nSources: {', '.join(result['sources'])}")
                print(f"Model: {result['model_used']}")
                print(f"Context chunks used: {result['context_chunks']}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    # Single query mode
    else:
        query = " ".join(sys.argv[1:])
        result = query_system.query(query)
        
        print(f"Query: {result['query']}")
        print(f"Response: {result['response']}")
        print(f"Sources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()