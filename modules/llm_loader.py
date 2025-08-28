# modules/llm_loader.py

import os
from langchain_openai import ChatOpenAI

def load_llm(model_name: str):
    """
    根据模型名加载对应的大语言模型
    支持：gpt-4, deepseek-v3, deepseek-r1
    """

    if model_name == "gpt-4":
        return ChatOpenAI(
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY")
        )

    elif model_name == "deepseek-v3":
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    elif model_name == "deepseek-r1":
        return ChatOpenAI(
            model="deepseek-reasoner",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    else:
        raise ValueError(f"暂不支持的模型类型：{model_name}")
