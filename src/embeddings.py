# Simple embeddings module using TF-IDF as a fallback when advanced embeddings are not available 

import json
import os
import pickle
from typing import List, Dict, Any, Optional
from collections import Counter
import math


class SimpleEmbeddings:
    """Simple TF-IDF based embeddings as a fallback"""
    
    def __init__(self):
        self.vocabulary = {}
        self.idf_scores = {}
        self.fitted = False
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        import re
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency"""
        tf = {}
        total_tokens = len(tokens)
        token_counts = Counter(tokens)
        
        for token, count in token_counts.items():
            tf[token] = count / total_tokens
            
        return tf
    
    def _compute_idf(self, documents: List[str]) -> Dict[str, float]:
        """Compute inverse document frequency"""
        doc_count = len(documents)
        word_doc_count = {}
        
        for doc in documents:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                word_doc_count[token] = word_doc_count.get(token, 0) + 1
        
        idf = {}
        for word, count in word_doc_count.items():
            idf[word] = math.log(doc_count / count)
            
        return idf
    
    def fit(self, documents: List[str]):
        """Fit the TF-IDF model on documents"""
        all_tokens = []
        
        for doc in documents:
            tokens = self._tokenize(doc)
            all_tokens.extend(tokens)
        
        # Build vocabulary
        unique_tokens = list(set(all_tokens))
        self.vocabulary = {token: idx for idx, token in enumerate(unique_tokens)}
        
        # Compute IDF scores
        self.idf_scores = self._compute_idf(documents)
        self.fitted = True
        
    def transform(self, text: str) -> List[float]:
        """Transform text to TF-IDF vector"""
        if not self.fitted:
            raise ValueError("Model must be fitted before transforming")
        
        tokens = self._tokenize(text)
        tf_scores = self._compute_tf(tokens)
        
        # Create vector
        vector = [0.0] * len(self.vocabulary)
        
        for token, tf in tf_scores.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                idf = self.idf_scores.get(token, 0)
                vector[idx] = tf * idf
                
        return vector
    
    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        """Fit the model and transform documents"""
        self.fit(documents)
        return [self.transform(doc) for doc in documents]


class VectorStore:
    """Simple vector storage using cosine similarity"""
    
    def __init__(self):
        self.vectors = []
        self.metadata = []
        self.embeddings_model = SimpleEmbeddings()
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store"""
        texts = [doc['text'] for doc in documents]
        
        # Fit and transform documents
        vectors = self.embeddings_model.fit_transform(texts)
        
        self.vectors.extend(vectors)
        self.metadata.extend(documents)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)
    
    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.vectors:
            return []
            
        query_vector = self.embeddings_model.transform(query)
        
        # Compute similarities
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_vector, doc_vector)
            similarities.append((similarity, i))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        
        results = []
        for similarity, idx in similarities[:k]:
            result = self.metadata[idx].copy()
            result['similarity'] = similarity
            results.append(result)
            
        return results
    
    def save(self, filepath: str):
        """Save the vector store to file"""
        data = {
            'vectors': self.vectors,
            'metadata': self.metadata,
            'embeddings_model': {
                'vocabulary': self.embeddings_model.vocabulary,
                'idf_scores': self.embeddings_model.idf_scores,
                'fitted': self.embeddings_model.fitted
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load the vector store from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.vectors = data['vectors']
        self.metadata = data['metadata']
        
        # Restore embeddings model
        model_data = data['embeddings_model']
        self.embeddings_model.vocabulary = model_data['vocabulary']
        self.embeddings_model.idf_scores = model_data['idf_scores']
        self.embeddings_model.fitted = model_data['fitted']


if __name__ == "__main__":
    # Test the vector store
    from document_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    chunks = processor.process_documents()
    
    print(f"Creating vector store with {len(chunks)} chunks")
    
    vector_store = VectorStore()
    vector_store.add_documents(chunks)
    
    # Test search
    query = "What is machine learning?"
    results = vector_store.similarity_search(query)
    
    print(f"\nSearch results for: '{query}'")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Source: {result['source']} (Similarity: {result['similarity']:.3f})")
        print(f"Text: {result['text'][:200]}...")