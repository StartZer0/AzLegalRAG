"""
LLM generation module for AzLegalRAG.
Uses Mistral-7B-Instruct via HuggingFace transformers.
"""

import gc
import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

try:
    from .config import LLM_MODEL, TOP_K
    from .embed import clear_gpu_memory
except ImportError:
    from config import LLM_MODEL, TOP_K
    from embed import clear_gpu_memory


# Azerbaijani legal assistant prompt
PROMPT_TEMPLATE = """You are an Azerbaijani legal assistant.
Answer the question based ONLY on the provided context.
Always cite sources with their e-qanun.az links.
If you don't know the answer, say so.

Context:
{context}

Question: {question}

Answer (respond in the same language as the question):"""


_llm_instance = None
_llm_pipeline = None


def get_llm(model_name=None, max_new_tokens=512, force_reload=False):
    """
    Initialize Mistral-7B-Instruct LLM.
    Uses singleton pattern to avoid reloading.
    """
    global _llm_instance, _llm_pipeline
    
    if _llm_instance is not None and not force_reload:
        print("Using cached LLM instance")
        return _llm_instance
    
    model_name = model_name or LLM_MODEL
    
    print(f"Loading LLM: {model_name}")
    print(f"GPU available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"GPU Memory Free: {mem_free / 1e9:.1f} GB")
    
    _llm_pipeline = pipeline(
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
    
    _llm_instance = HuggingFacePipeline(pipeline=_llm_pipeline)
    print("LLM loaded successfully!")
    
    return _llm_instance


def unload_llm():
    """Unload LLM to free GPU memory."""
    global _llm_instance, _llm_pipeline
    
    print("Unloading LLM...")
    
    if _llm_pipeline is not None:
        del _llm_pipeline
        _llm_pipeline = None
    
    if _llm_instance is not None:
        del _llm_instance
        _llm_instance = None
    
    clear_gpu_memory()


def normalize_text(text):
    """Fix split Azerbaijani characters from bad PDF extraction."""
    if not text:
        return ""
    import unicodedata
    import re
    
    text = unicodedata.normalize('NFC', text)
    
    # Azerbaijani alphabet (Latin + special chars)
    az_chars = r'[a-zA-ZəüöşçğıƏÜÖŞÇĞI]'
    
    # Aggressively join letters separated by newlines
    for _ in range(10):
        prev = text
        text = re.sub(f'({az_chars})\\n({az_chars})', r'\1\2', text)
        if text == prev:
            break
    
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def format_docs(docs):
    """Format retrieved documents as context string with normalized text."""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        content = normalize_text(doc.page_content)
        context_parts.append(f"[{i}] Source: {source}\n{content}")
    return "\n\n".join(context_parts)


def create_rag_chain(vectorstore, llm=None):
    """
    Create RAG chain with retrieval and generation using LCEL.
    
    Args:
        vectorstore: Chroma vectorstore instance
        llm: LangChain LLM (will be created if None)
    
    Returns:
        RAG chain
    """
    if llm is None:
        llm = get_llm()
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3}
    )
    
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # LCEL chain (LangChain Expression Language)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever


def ask_question(chain, retriever, question):
    """
    Ask a question using the RAG chain.
    
    Args:
        chain: RAG chain
        retriever: Document retriever
        question: User question string
    
    Returns:
        Dict with 'result' and 'source_documents'
    """
    # Get answer
    result = chain.invoke(question)
    
    # Get source documents
    source_docs = retriever.invoke(question)
    
    return {
        "result": result,
        "source_documents": source_docs
    }


if __name__ == "__main__":
    print("Testing LLM loading...")
    llm = get_llm()
    print("LLM loaded successfully!")
    unload_llm()
    print("LLM unloaded successfully!")
