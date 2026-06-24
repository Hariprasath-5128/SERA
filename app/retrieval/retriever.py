import logging
from typing import List, Dict, Any
from app.db.chroma_client import ChromaClient
from app.ingestion.embedder import Embedder
from app.config import DEFAULT_TOP_K

logger = logging.getLogger(__name__)

class Retriever:
    @classmethod
    def search(cls, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Embeds the user query and performs a Cosine Similarity search
        against the ChromaDB vector database.
        
        Args:
            query (str): The user's question (e.g. "What are the symptoms of Mabry syndrome?")
            top_k (int): Number of most mathematically relevant chunks to return.
            
        Returns:
            List of dictionaries containing the chunk text and its metadata.
        """
        if not query.strip():
            return []
            
        # 1. Embed the query using the exact same BGE-M3 model we used for ingestion
        model = Embedder.get_model()
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].tolist()
        
        # 2. Query ChromaDB for the closest vectors
        collection = ChromaClient.get_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 3. Parse and format the results
        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] # Lower distance = Higher Cosine Similarity
            
            for i in range(len(ids)):
                formatted_results.append({
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i]
                })
                
        logger.info(f"Retrieved {len(formatted_results)} chunks for query: '{query}'")
        return formatted_results

if __name__ == "__main__":
    print("=========================================")
    print("     SERA Retriever Interactive Test     ")
    print("=========================================")
    print("Sample question to try: 'What are the symptoms of Mabry syndrome?'\n")
    
    while True:
        query = input("Enter your medical question (or 'q' to quit): ").strip()
        if query.lower() in ['q', 'quit', 'exit']:
            break
            
        if not query:
            continue
            
        print(f"\n[Retrieving top chunks for: '{query}']...")
        results = Retriever.search(query, top_k=3)
        
        if not results:
            print("No results found in ChromaDB.")
            continue
            
        for i, res in enumerate(results, 1):
            print(f"\n--- Result {i} (Distance: {res['distance']:.4f}) ---")
            print(f"Section: {res['metadata'].get('section', 'N/A')}")
            print(f"Source: {res['metadata'].get('source_url', 'N/A')}")
            print(f"Text snippet: {res['text'][:250]}...\n")
