import os
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from .rag import SimpleRAG

load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

class ParkingAssistant:
    def __init__(self, rag: SimpleRAG) -> None:
        self.rag = rag
        if HF_API_KEY:
            self.client = InferenceClient(api_key=HF_API_KEY)
        else:
            self.client = None

    def answer(self, question: str, context: str = "") -> dict:
        normalized_question = (question or "").strip()
        if not normalized_question:
            return {
                "answer": "Veuillez saisir une question métier (accès, tarif, sortie, refus...).",
                "sources": [],
                "highlights": [],
            }

        query = normalized_question if not context else f"{normalized_question} {context}"
        docs = self.rag.retrieve(query, top_k=3)

        if not docs:
            return {
                "answer": "Je ne trouve pas de règle interne correspondante dans la base documentaire. Reformulez avec des mots-clés comme: blacklist, franchise, visiteur, sortie, horaires conditionnels.",
                "sources": [],
                "highlights": [],
            }

        highlights = []
        snippets = []
        for d in docs:
            highlights.append(f"{d['source']} · score {d['score']:.2f}")
            snippets.append(d['snippet'])

        context_text = "\n---\n".join(snippets)

        if self.client:
            prompt = f"""Tu es un assistant IA spécialisé dans les règlements du parking SmartPark. 
Utilise UNIQUEMENT le contexte documentaire ci-dessous pour répondre à la question de l'utilisateur.
Réponds en français, de manière claire, concise et professionnelle. Si la réponse n'est pas dans le document, dis-le poliment.

Documents de référence:
{context_text}

Informations supplémentaires (contexte du véhicule): {context}

Question de l'utilisateur: {normalized_question}

Réponse:"""

            try:
                # Using Mistral 7B / Qwen 2.5 72B for excellent multilingual text generation
                model_id = "Qwen/Qwen2.5-72B-Instruct"
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.3
                )
                final_answer = response.choices[0].message.content.strip()
            except Exception as e:
                # Fallback to pure snippet if API fails
                final_answer = f"⚠️ Erreur de l'API IA: {e}\n\nSynthèse brute depuis les documents:\n{docs[0]['snippet']}"
        else:
            final_answer = f"⚠️ L'IA est désactivée (Clé API Hugging Face manquante). Voici l'extrait brut :\n{docs[0]['snippet']}"
            
        return {
            "answer": final_answer,
            "sources": [d["source"] for d in docs],
            "highlights": highlights,
        }

