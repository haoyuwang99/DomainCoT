from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_openai import ChatOpenAI


def get_hugging_face_llm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # trustworthy = model_n 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", # Use bfloat16 for efficiency if supported
        device_map="auto",
        trust_remote_code=True # Required for Qwen models
    )
    text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=text_gen)
    return llm

OPENAI_KEY = ""
with open("llmkey") as f:
    OPENAI_KEY = f.read()
    
def get_openai_llm(model_name = "gpt-4.1-mini", key=OPENAI_KEY, temperature=0):
    llm = ChatOpenAI(model = model_name,
                 api_key=key,
                 temperature=temperature)
    return llm

