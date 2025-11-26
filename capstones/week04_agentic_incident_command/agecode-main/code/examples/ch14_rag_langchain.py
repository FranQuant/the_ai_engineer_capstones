# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Optional snapshot: LangChain + Chroma retriever for Chapter 14.

Requires optional deps: langchain, langchain-community, chromadb,
sentence-transformers.
"""

from __future__ import annotations  # Future annotations.

from langchain_community.vectorstores import Chroma  # Vector store.
from langchain_community.embeddings.sentence_transformer import (  # Embeddings.
    SentenceTransformerEmbeddings,
)


# tag::ch14_rag_langchain[]
def demo() -> None:  # Minimal LC + Chroma.
    texts = [  # Three docs.
        "Alpha launched in 2022 with a focus on simplicity.",  # Doc 1.
        "Key benefit: transparency in logs and short audits.",  # Doc 2.
        "Beta emphasized speed over explainability in 2021.",  # Doc 3.
    ]
    metadatas = [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}]  # Ids.
    embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  # Model.
    db = Chroma.from_texts(
        texts=texts,
        embedding=embed,
        metadatas=metadatas,
    )  # Index.
    retriever = db.as_retriever(search_kwargs={"k": 2})  # Top‑2 retriever.
    hits = retriever.invoke("Alpha launch year")  # Query.
    print([(h.metadata.get("id"), h.page_content[:30]) for h in hits])  # Show ids.


if __name__ == "__main__":
    demo()
# end::ch14_rag_langchain[]
