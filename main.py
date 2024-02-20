from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import (
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

from dotenv import load_dotenv
import os
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from langchain.chains import LLMChain


load_dotenv()


openai_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    api_key=openai_key,
    verbose=True,
)

memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory("message.json"),
    memory_key="messages",
    return_messages=True,
    llm=chat,
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)

while True:
    content = input(">> ")

    result = chain({"content": content})

    if "text" in result:
        # print(memory)
        print(result["text"])
    else:
        print("Error: Unable to retrieve text from result.")
