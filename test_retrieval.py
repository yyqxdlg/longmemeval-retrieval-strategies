from retrieval import (
    StellaRetriever,
    retrieve_recency,
    retrieve_semantic,
    retrieve_hybrid,
)


def show(title, items):
    print(f"\n{title}")
    for item in items:
        score_parts = []

        for key in (
            "semantic_cosine",
            "semantic_score",
            "recency_score",
            "hybrid_score",
        ):
            if key in item:
                score_parts.append(f"{key}={item[key]:.4f}")

        print(
            item["id"],
            "|",
            item["text"].replace("\n", " "),
            "|",
            ", ".join(score_parts),
        )


def main():
    memories = [
        {
            "id": "m1",
            "text": (
                "User: My favorite drink is green tea.\n"
                "Assistant: I will remember that."
            ),
            "recency_order": 1,
        },
        {
            "id": "m2",
            "text": (
                "User: I am planning a trip to London.\n"
                "Assistant: London has many interesting places."
            ),
            "recency_order": 2,
        },
        {
            "id": "m3",
            "text": (
                "User: I recently started drinking coffee.\n"
                "Assistant: That is good to know."
            ),
            "recency_order": 3,
        },
    ]

    query = "What drink does the user like?"

    retriever = StellaRetriever()
    memory_embeddings = retriever.encode_memories(memories)

    recency = retrieve_recency(memories, k=2)

    semantic = retrieve_semantic(
        query,
        memories,
        memory_embeddings,
        retriever,
        k=2,
    )

    hybrid = retrieve_hybrid(
        query,
        memories,
        memory_embeddings,
        retriever,
        k=2,
    )

    show("RECENCY", recency)
    show("SEMANTIC", semantic)
    show("HYBRID", hybrid)


if __name__ == "__main__":
    main()
