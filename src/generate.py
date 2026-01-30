"""
LLM generation module for AzLegalRAG.
Uses Mistral-7B-Instruct via HuggingFace transformers.
Implements sequential loading - loads after embedding model is unloaded.
"""

import gc
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

try:
    from .config import LLM_MODEL, TOP_K
    from .embed import clear_gpu_memory, unload_embeddings
except ImportError:
    from config import LLM_MODEL, TOP_K
    from embed import clear_gpu_memory, unload_embeddings


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
    
    Args:
        model_name: HuggingFace model ID
        max_new_tokens: Maximum tokens to generate
        force_reload: Force reload even if already loaded
    
    Returns:
        LangChain-compatible LLM
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
    
    template = prompt_template or PROMPT_TEMPLATE
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
    unload_llm()
    print("LLM unloaded successfully!")
