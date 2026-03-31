# -*- coding: utf-8 -*-

import os
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGPipeline:

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0)
        self.vectorstore = None

    def load_documents(self, paths: List[str]):
        docs = []
        for path in paths:
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path)
            docs.extend(loader.load())
        return docs

    def ingest_documents(self, paths: List[str]) -> int:
        docs = self.load_documents(paths)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(docs)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        return len(chunks)

    def query(self, question: str):
        retriever = self.vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(question)

        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer using ONLY this context:

        {context}

        Question: {question}
        """

        answer = self.llm.predict(prompt)

        return {
            "answer": answer,
            "sources": [{"content": doc.page_content[:200]} for doc in docs],
        }

    def stream_query(self, question: str):
        result = self.query(question)
        for token in result["answer"].split():
            yield token