from datetime import datetime

from .rag import SimpleRAG


class ParkingAssistant:
    def __init__(self, rag: SimpleRAG) -> None:
        self.rag = rag

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

        top = docs[0]
        highlights = []
        for d in docs:
            highlights.append(f"{d['source']} · score {d['score']:.2f}")

        answer_lines = [
            "Voici la meilleure réponse basée sur les documents internes.",
            f"Question: {normalized_question}",
        ]

        if context:
            answer_lines.append(f"Contexte pris en compte: {context}")

        answer_lines.extend(
            [
                "",
                "Synthèse:",
                top["snippet"],
                "",
                f"Référence principale: {top['source']}",
                f"Horodatage: {datetime.now().isoformat(timespec='seconds')}",
            ]
        )

        return {
            "answer": "\n".join(answer_lines),
            "sources": [d["source"] for d in docs],
            "highlights": highlights,
        }
