#import os
import json
import requests
import traceback

from model_configurations import get_model_configuration

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import base64
from mimetypes import guess_type
from langchain_core.messages import HumanMessage, SystemMessage

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    llm = AzureChatOpenAI(
        model = gpt_config['model_name'],
        deployment_name = gpt_config['deployment_name'],
        openai_api_key = gpt_config['api_key'],
        openai_api_version = gpt_config['api_version'],
        azure_endpoint = gpt_config['api_base'],
        temperature= gpt_config['temperature']
    )

    system = """
            You are a helpful assistant.
            Please respond in JSON format.
            The top-level key must be 'Result', and its value must be a list of objects.
            Each object should contain two keys: 'date' (the date of the holiday) and 'name' (the name of the holiday).
            """
    try:
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Answer the question based on the below description and only use English or traditional Chinese. Please respond in JSON format."),
                ("human", f"Question: {question} \n "),
            ]
        )

        json_llm = llm.bind(response_format={"type": "json_object"})
        
        rag_chain = LLMChain(
            llm=json_llm,
            prompt=answer_prompt,
            output_parser=StrOutputParser()
        )

        # RAG generation
        answer = rag_chain.invoke({"question": question})
        print(answer)
        text_content = answer.get('text')

        return text_content
    except Exception as e:
        traceback_info = traceback.format_exc()
        print(traceback_info)

# Define a tool function to query Taiwan's holidays for a specific month
def query_calendar(month: int) -> str:
    """
    Query Taiwan's holidays for a specific month and return the results as a string.
    """
    try:
        api_key = 'XCN0zMZN7qmPMA1W37i6fHWd94NibOqQ'  # Replace with your actual API key
        country = 'TW'
        year = 2024  # Adjust the year as needed
        url = f'https://calendarific.com/api/v2/holidays?api_key={api_key}&country={country}&year={year}&month={month}'

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            holidays = data.get('response', {}).get('holidays', [])
            result_list = [
                f"{holiday['date']['iso'][5:10]}: {holiday['name']}"
                for holiday in holidays
            ]
            return "\n".join(result_list)
        else:
            return f"Unable to fetch data, HTTP status code: {response.status_code}"
    except Exception as e:
        return f"Tool invocation failed: {str(e)}"

def generate_hw02(question):
    llm = AzureChatOpenAI(
        model = gpt_config['model_name'],
        deployment_name = gpt_config['deployment_name'],
        openai_api_key = gpt_config['api_key'],
        openai_api_version = gpt_config['api_version'],
        azure_endpoint = gpt_config['api_base'],
        temperature= gpt_config['temperature']
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", (
                "You are a helpful assistant capable of answering holiday-related questions. "
                "When returning the result, please format it as JSON and use 'Result' as the key for the list of holidays. "
                "Do not use 'holidays' as the key."
            )),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{input}")
        ]
    )

    # Create a list of tools to be used by the agent
    tools = [
        Tool(
            name="query_calendar",
            func=query_calendar,
            description=(
                "Use this tool to query Taiwan's holidays for a specific month. "
                "Input should be an integer representing the month (e.g., 1 for January)."
            )
        )
    ]

    llm_json = llm.bind(response_format={"type": "json_object"})
    # Create an agent using the Azure OpenAI language model, tools, and prompt
    agent = create_openai_functions_agent(
        llm=llm_json,
        tools=tools,
        prompt=prompt
    )

    # Create an AgentExecutor to manage the agent's execution
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": question})
    print(f'Result:{response}')
    output_content = response.get('output', '{}')

    return output_content

def generate_hw03(question1, question2):
    llm = AzureChatOpenAI(
        model = gpt_config['model_name'],
        deployment_name = gpt_config['deployment_name'],
        openai_api_key = gpt_config['api_key'],
        openai_api_version = gpt_config['api_version'],
        azure_endpoint = gpt_config['api_base'],
        temperature= gpt_config['temperature']
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", (
                "You are a helpful assistant capable of answering holiday-related questions. "
                "For the user's query:\n"
                "- If the query is about holidays for a specific month, return a JSON object with the key 'Result' containing the list of holidays.\n"
                "- If the query is about whether a specific holiday should be added, return a JSON object with the keys 'Result':\n"
                "  - 'add': A boolean indicating whether the holiday should be added.\n"
                "  - 'reason': A string explaining your decision."
            )),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{input}")
        ]
    )

    # Create a list of tools to be used by the agent
    tools = [
        Tool(
            name="query_calendar",
            func=query_calendar,
            description=(
                "Use this tool to query Taiwan's holidays for a specific month. "
                "Input should be an integer representing the month (e.g., 1 for January)."
            )
        )
    ]

    llm_json = llm.bind(response_format={"type": "json_object"})

    store = {}
    # Function to get or create chat message history based on session_id
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    # Create an agent using the Azure OpenAI language model, tools, and prompt
    agent = create_openai_functions_agent(
        llm=llm_json,
        tools=tools,
        prompt=prompt
    )

    # Create an AgentExecutor to manage the agent's execution
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    response = agent_with_chat_history.invoke(
            {"input": question1},
            config={"configurable": {"session_id": "<foo>"}},
    )
    print(f'response1:{response}')
    response = agent_with_chat_history.invoke(
            {"input": question2},
            config={"configurable": {"session_id": "<foo>"}},
    )
    print(f'response2:{response}')
    output_content = response.get('output', '{}')
    
    return output_content

def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    return f"data:{mime_type};base64,{base64_encoded_data}"

def generate_hw04(question):
    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

    image_data_url = local_image_to_data_url("baseball.png")

    system_message = SystemMessage(
        content=("""
        Please strictly follow the JSON format below to return the result without including any extra content:
        {
            "Result": {
                    "score": <parsed score>
            }
        }
        Ensure that `score` is parsed from the content of the image provided by the user.
        """
        )
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    )
    llm_json = llm.bind(response_format={"type": "json_object"})
    response = llm_json.invoke([system_message, message])
    print(response.content)

    return response.content

def generate_hw05(question):
    pass

if __name__ == '__main__':
    try:
        # hw01
        question = '2024年台灣10月紀念日有哪些?'
        answer = generate_hw01(question)
        parsed_data = json.loads(answer)
        result_count = len(parsed_data["Result"])
        print(f'HW01:{result_count}')

        # hw02
        question = "What are the holidays in Taiwan for October?"
        answer = generate_hw02(question)
        parsed_data = json.loads(answer)
        result_count = len(parsed_data["Result"])
        print(f'HW02:{result_count}')

        # hw03
        question1 = "What are the holidays in Taiwan for October?"
        new_holiday = {"date": "10-31", "name": "Chiang Kai-shek's Birthday Memorial Day"}
        question2 = (
            f"Based on the previous list of holidays, is this holiday included in the list for that month?"
            f"{json.dumps(new_holiday, ensure_ascii=False)}"
        )
        answer = generate_hw03(question1, question2)
        parsed_data = json.loads(answer)
        result_count = parsed_data["Result"]["add"]
        print(f'HW03:{result_count}')

        # hw04
        question = '請問中華台北的積分是多少'
        answer = generate_hw04(question)
        parsed_data = json.loads(answer)
        result_count = parsed_data["Result"]["score"]
        print(f'HW04:{result_count}')

        print('End')
    except Exception as e:
        print(e)


