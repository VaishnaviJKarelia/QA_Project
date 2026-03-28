import importlib

import main


def test_pdf_search_tool(monkeypatch):
    # Dummy vector store exposing as_retriever
    class DummyVectorStore:
        def as_retriever(self, search_kwargs=None):
            return "dummy_retriever"

    # Mock RetrievalQA.from_chain_type to return a fake chain with a run method
    class MockQAChain:
        def run(self, query: str) -> str:
            return f"MOCK_ANSWER: {query}"

    monkeypatch.setattr(main, "RetrievalQA", main.RetrievalQA)
    monkeypatch.setattr(main.RetrievalQA, "from_chain_type", lambda llm, chain_type, retriever: MockQAChain())

    # Patch the ChatGoogleGenerativeAI constructor used in create_pdf_search_tool
    monkeypatch.setattr(main, "ChatGoogleGenerativeAI", lambda *args, **kwargs: "mock_llm")

    tool = main.create_pdf_search_tool(DummyVectorStore())
    result = tool.func("What is in the document?")

    assert isinstance(result, str)
    assert "MOCK_ANSWER:" in result
