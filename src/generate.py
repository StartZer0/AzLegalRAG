"""
LLM generation module for AzLegalRAG.
Uses Mistral-7B-Instruct via HuggingFace transformers.
"""

import torch
from transformers import pipeline, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

try:
    from .config import LLM_MODEL, TOP_K
except ImportError:
    from config import LLM_MODEL, TOP_K


# Azerbaijani legal assistant prompt
PROMPT_TEMPLATE = """Siz Azerbaycan huquq meseleleri uzre komekcisisiniz. 
Suala yalniz verilmis kontekst esasinda cavab verin.
Menbeleri e-qanun.az linki ile gosterin.

Kontekst:
{context}

Sual: {question}

Azerbaycan dilinde cavab:"""

PROMPT_TEMPLATE_EN = """You are an Azerbaijani legal assistant.
Answer the question based ONLY on the provided context.
Always cite sources with their e-qanun.az links.

Context:
{context}

Question: {question}

Answer (in Azerbaijani or the language of the question):"""


def get_llm(model_name=None, max_new_tokens=512):
    """
    Initialize Mistral-7B-Instruct LLM.
    
    Args:
        model_name: HuggingFace model ID
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        LangChain-compatible LLM
    """
    model_name = model_name or LLM_MODEL
    
    print(f"Loading LLM: {model_name}")
    print(f"GPU available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    return HuggingFacePipeline(pipeline=pipe)


def create_rag_chain(vectorstore, llm=None, prompt_template=None):
    """
    Create RAG chain with retrieval and generation.
    
    Args:
        vectorstore: Chroma vectorstore instance
        llm: LangChain LLM (will be created if None)
        prompt_template: Custom prompt template
    
    Returns:
        RetrievalQA chain
    """
    if llm is None:
        llm = get_llm()
    
    template = prompt_template or PROMPT_TEMPLATE_EN
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return chain


def ask_question(chain, question):
    """
    Ask a question using the RAG chain.
    
    Args:
        chain: RetrievalQA chain
        question: User question string
    
    Returns:
        Dict with 'result' and 'source_documents'
    """
    result = chain({"query": question})
    return result


if __name__ == "__main__":
    print("Testing LLM loading...")
    llm = get_llm()
    print("LLM loaded successfully!")
