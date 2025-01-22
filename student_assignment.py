import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    output_parser = JsonOutputParser()
    answer_format = "{{\"Result\": [{{\"date\": \"2024-10-10\",\"name\": \"國慶日\"}}]}}"
    prompt_template = " \n請依據問題語系回答: {query}，僅使用正確的Json格式回應 ,格式範例如下：" + answer_format
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query"]
    )

    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    chain = prompt | llm
    raw_output = chain.invoke({"query": question})
    parsed_output = output_parser.invoke(raw_output)
    result = json.dumps(parsed_output, ensure_ascii=False, indent=2)
    print(result)
    return result

def generate_hw02(question):
    pass

def generate_hw03(question2, question3):
    pass

def generate_hw04(question):
    pass

def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response