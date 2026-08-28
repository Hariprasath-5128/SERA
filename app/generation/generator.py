import os
import logging
from typing import List, Dict, Any
from openai import OpenAI
from app.config import GENERATOR_LLM_MODEL

logger = logging.getLogger(__name__)

class Generator:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            # Assumes OPENAI_API_KEY is loaded in environment
            api_key = os.getenv("OPENAI_API_KEY", "dummy-key-for-local-models")
            
            from app.config import LLM_BASE_URL
            cls._client = OpenAI(
                api_key=api_key,
                base_url=LLM_BASE_URL  # If None, it defaults to standard OpenAI endpoint
            )
        return cls._client

    @classmethod
    def generate_answer(cls, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Synthesizes a final conversational answer using an LLM, heavily grounded
        by the raw chunks retrieved from ChromaDB.
        """
        if not retrieved_chunks:
            return "I couldn't find any relevant medical information in the database to answer your question."
            
        client = cls.get_client()
        
        # 1. Compile the context window
        context_text = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk['text']
            context_text += f"{text}\n\n"

        system_prompt = (
            "You are a precise medical expert. Answer the question using ONLY the facts "
            "provided in the background context. Extract and state the most relevant medical "
            "facts directly. Do not add introductions, conclusions, or phrases like "
            "'Based on the context' or 'According to'. Do not repeat the question. "
            "State medical facts concisely and precisely."
        )
        
        user_prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer (medical facts only, no filler):"
        )
        
        # 3. Call the LLM
        try:
            logger.info(f"Generating answer using {GENERATOR_LLM_MODEL}...")
            client = Generator.get_client()
            response = client.chat.completions.create(
                model=GENERATOR_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"An error occurred while communicating with the LLM: {str(e)}"

if __name__ == "__main__":
    import sys
    # Hack to load .env token for standalone testing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    sys.path.insert(0, project_root)
    
    from dotenv import load_dotenv
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
    
    from app.retrieval.retriever import Retriever
    
    print("=========================================")
    print("     SERA End-to-End RAG Interactive     ")
    print("=========================================")
    
    while True:
        query = input("\nEnter your medical question (or 'q' to quit): ").strip()
        if query.lower() in ['q', 'quit', 'exit']:
            break
            
        if not query:
            continue
            
        print(f"\n[1] Retrieving top chunks from ChromaDB...")
        results = Retriever.search(query, top_k=3)
        
        if not results:
            print("No results found in ChromaDB.")
            continue
            
        print(f"[2] Synthesizing answer with {GENERATOR_LLM_MODEL}...\n")
        answer = Generator.generate_answer(query, results)
        
        print("🤖 SERA:")
        print(answer)
        print("\n" + "-"*40)
