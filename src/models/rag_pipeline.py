"""
Retrieval-Augmented Generation Pipeline for F1 Insights
Retrieves relevant news and generates natural language explanations
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass
import openai
from datetime import datetime


@dataclass
class Document:
    """Represents a news article or text document"""
    id: str
    title: str
    content: str
    source: str
    published_at: datetime
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class DocumentRetriever:
    """Retrieves relevant documents using semantic search"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents: List[Document] = []
        self.dimension = self.encoder.get_sentence_embedding_dimension()
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the retrieval index"""
        
        # Generate embeddings
        texts = [f"{doc.title} {doc.content}" for doc in documents]
        embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True
        )
        
        # Store embeddings with documents
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
            self.documents.append(doc)
        
        # Build FAISS index
        embeddings_array = np.array([doc.embedding for doc in self.documents])
        
        # Use IndexFlatIP for cosine similarity (after normalization)
        faiss.normalize_L2(embeddings_array)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_array)
        
        print(f"Added {len(documents)} documents to index")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve most relevant documents for a query
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            min_score: Minimum similarity score threshold
            
        Returns:
            List of (document, score) tuples
        """
        
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])[0]
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Filter by minimum score and return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_score and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save_index(self, path: Path):
        """Save index and documents to disk"""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(path / "faiss_index.bin"))
        
        # Save documents
        docs_data = []
        for doc in self.documents:
            docs_data.append({
                'id': doc.id,
                'title': doc.title,
                'content': doc.content,
                'source': doc.source,
                'published_at': doc.published_at.isoformat(),
                'embedding': doc.embedding.tolist() if doc.embedding is not None else None,
                'metadata': doc.metadata
            })
        
        with open(path / "documents.json", 'w') as f:
            json.dump(docs_data, f)
    
    def load_index(self, path: Path):
        """Load index and documents from disk"""
        
        # Load FAISS index
        index_path = path / "faiss_index.bin"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        
        # Load documents
        docs_path = path / "documents.json"
        if docs_path.exists():
            with open(docs_path, 'r') as f:
                docs_data = json.load(f)
            
            self.documents = []
            for doc_dict in docs_data:
                doc = Document(
                    id=doc_dict['id'],
                    title=doc_dict['title'],
                    content=doc_dict['content'],
                    source=doc_dict['source'],
                    published_at=datetime.fromisoformat(doc_dict['published_at']),
                    embedding=np.array(doc_dict['embedding']) if doc_dict['embedding'] else None,
                    metadata=doc_dict.get('metadata')
                )
                self.documents.append(doc)


class F1ExplanationGenerator:
    """Generates natural language explanations using LLM"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo"
    ):
        if api_key:
            openai.api_key = api_key
        self.model = model
    
    def generate_explanation(
        self,
        prediction: Dict,
        context_documents: List[Tuple[Document, float]],
        driver_name: str,
        max_length: int = 500
    ) -> str:
        """
        Generate explanation for a prediction using retrieved context
        
        Args:
            prediction: Dictionary with prediction values
            context_documents: Retrieved relevant documents
            driver_name: Name of the driver
            max_length: Maximum response length
            
        Returns:
            Natural language explanation
        """
        
        # Build context from documents
        context_text = self._build_context(context_documents)
        
        # Build prediction summary
        pred_summary = self._build_prediction_summary(prediction, driver_name)
        
        # Create prompt
        prompt = f"""You are an expert F1 analyst. Based on the following prediction and recent news context, provide a clear and insightful explanation of what might influence {driver_name}'s performance.

Prediction:
{pred_summary}

Recent Context:
{context_text}

Provide a concise analysis (max {max_length} words) explaining:
1. Key factors affecting the prediction
2. How recent events/news might impact performance
3. Any risks or opportunities to watch for

Keep the tone professional but accessible."""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert F1 analyst providing insightful performance analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length * 2,  # Approximate token count
                temperature=0.7
            )
            
            return response.choices[0].message['content'].strip()
        
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return self._generate_fallback_explanation(prediction, driver_name, context_documents)
    
    def _build_context(self, documents: List[Tuple[Document, float]]) -> str:
        """Build context string from retrieved documents"""
        
        context_parts = []
        for doc, score in documents[:3]:  # Use top 3 documents
            context_parts.append(
                f"[{doc.source}, {doc.published_at.strftime('%Y-%m-%d')}]\n"
                f"{doc.title}\n{doc.content[:300]}..."
            )
        
        return "\n\n".join(context_parts)
    
    def _build_prediction_summary(self, prediction: Dict, driver_name: str) -> str:
        """Build prediction summary text"""
        
        summary = f"{driver_name} Performance Forecast:\n"
        
        if 'predicted_points' in prediction:
            summary += f"- Expected Points: {prediction['predicted_points']:.1f}\n"
        
        if 'predicted_position' in prediction:
            summary += f"- Expected Position: P{int(prediction['predicted_position'])}\n"
        
        if 'confidence_80_lower' in prediction and 'confidence_80_upper' in prediction:
            summary += f"- 80% Confidence Range: {prediction['confidence_80_lower']:.1f} - {prediction['confidence_80_upper']:.1f} points\n"
        
        return summary
    
    def _generate_fallback_explanation(
        self,
        prediction: Dict,
        driver_name: str,
        context_documents: List[Tuple[Document, float]]
    ) -> str:
        """Generate basic explanation without LLM"""
        
        explanation = f"Based on recent performance data, {driver_name} is predicted to score "
        explanation += f"{prediction.get('predicted_points', 'N/A')} points. "
        
        if context_documents:
            explanation += f"\n\nRecent developments:\n"
            for doc, _ in context_documents[:2]:
                explanation += f"- {doc.title}\n"
        
        return explanation


class F1RAGPipeline:
    """Complete RAG pipeline for F1 insights"""
    
    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        generator: Optional[F1ExplanationGenerator] = None,
        corpus_path: Optional[Path] = None
    ):
        self.retriever = retriever or DocumentRetriever()
        self.generator = generator or F1ExplanationGenerator()
        
        if corpus_path and corpus_path.exists():
            self.retriever.load_index(corpus_path)
    
    def add_news_corpus(self, news_df: pd.DataFrame):
        """Add news articles to the retrieval corpus"""
        
        documents = []
        for idx, row in news_df.iterrows():
            doc = Document(
                id=f"news_{idx}",
                title=row['title'],
                content=row.get('content', row.get('description', '')),
                source=row['source'],
                published_at=pd.to_datetime(row['published_at']),
                metadata={
                    'url': row.get('url'),
                    'category': row.get('category')
                }
            )
            documents.append(doc)
        
        self.retriever.add_documents(documents)
    
    def explain_prediction(
        self,
        driver_name: str,
        prediction: Dict,
        context_query: Optional[str] = None,
        top_k_docs: int = 5
    ) -> Dict[str, any]:
        """
        Generate comprehensive explanation for a prediction
        
        Args:
            driver_name: Name of the driver
            prediction: Prediction dictionary
            context_query: Optional custom query for retrieval
            top_k_docs: Number of documents to retrieve
            
        Returns:
            Dictionary with explanation and retrieved sources
        """
        
        # Build retrieval query
        if context_query is None:
            context_query = f"{driver_name} Formula 1 performance news injury team"
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(context_query, top_k=top_k_docs)
        
        # Generate explanation
        explanation = self.generator.generate_explanation(
            prediction,
            retrieved_docs,
            driver_name
        )
        
        # Format sources
        sources = [
            {
                'title': doc.title,
                'source': doc.source,
                'date': doc.published_at.strftime('%Y-%m-%d'),
                'relevance_score': float(score),
                'url': doc.metadata.get('url') if doc.metadata else None
            }
            for doc, score in retrieved_docs
        ]
        
        return {
            'explanation': explanation,
            'sources': sources,
            'num_documents_retrieved': len(retrieved_docs)
        }
    
    def save_corpus(self, path: Path):
        """Save the document corpus"""
        self.retriever.save_index(path)
    
    def batch_explain(
        self,
        predictions: List[Dict],
        driver_names: List[str]
    ) -> List[Dict]:
        """Generate explanations for multiple predictions"""
        
        results = []
        for pred, name in zip(predictions, driver_names):
            result = self.explain_prediction(name, pred)
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Example usage
    
    # Initialize pipeline
    rag = F1RAGPipeline()
    
    # Load news data
    news_df = pd.read_csv('data/raw/f1_news.csv')
    rag.add_news_corpus(news_df)
    
    # Make prediction (example)
    prediction = {
        'predicted_points': 18.5,
        'predicted_position': 3,
        'confidence_80_lower': 12.0,
        'confidence_80_upper': 25.0
    }
    
    # Generate explanation
    result = rag.explain_prediction(
        driver_name="Max Verstappen",
        prediction=prediction
    )
    
    print("Explanation:", result['explanation'])
    print("\nSources used:")
    for source in result['sources']:
        print(f"- {source['title']} ({source['source']}, {source['date']})")
    
    # Save corpus for future use
    rag.save_corpus(Path('data/news_corpus'))
