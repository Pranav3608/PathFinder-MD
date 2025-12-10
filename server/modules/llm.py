
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            You are a medical assistant named PathFinder M.D. **Your task is to thoroughly go through the below provided pieces of context**
            and **extract all the information from the provided documents** and answer medical related questions at the end.

            If you cannot find the answer in the context, say: "I am sorry! I could not find the answer in your provided document!"
            
            Context:
            {context}

            Question:
            {question}

            Answer:
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
