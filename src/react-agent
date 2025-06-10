import httpx
from langchain_core.tools import tool
import mlflow
import random

from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from langchain.agents import AgentExecutor, create_react_agent
import os

# ML Flow Configurations

mlflow.set_tracking_uri(uri="http://localhost:5000/")
mlflow.set_experiment("agenttest")
mlflow.langchain.autolog()

# Creating a Groq Chat model

httpx_client = httpx.Client(verify=False)
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"), model="gemma2-9b-it", http_client=httpx_client
)

# Creating tools

@tool("get_weather", description="Get the current weather for a given city")
def get_weather(city: str) -> str:
    presets = ["sunny", "rainy", "cloudy", "snowy"]
    weather = random.choice(presets)
    return f"It's {weather} in {city}!"


@tool(
    "get_temperature",
    description="Get temperature for a given city. The temperature is in celsius",
)
def get_temperature(city: str) -> int:
    presets = [25, 30, 35, 40, 45]
    temperature = random.choice(presets)
    return temperature


@tool(
    "get_drink",
    description="Recommended drink for a perticular weather and temperature",
)
def get_drink(weather: str, temperature: int = None) -> int:
    tempweather = ""
    temptemperature = 0
    if "," in weather:
        tempweather = weather.split(",")[0].strip()
        temptemperature = int(weather.split(",")[1].strip())
    else:
        tempweather = weather
        temptemperature = temperature
    drink = "coffee"
    if "sunny" in tempweather.lower() and temptemperature >= 35:
        drink = "fruit juice"
    elif "snowy" in tempweather.lower() and temptemperature <= 30:
        drink = "hot chocolate"
    elif "rainy" in tempweather.lower():
        drink = "green tea"
    return drink


@tool(
    "get_snacks",
    description="Recommended snack based on drink recommendation",
)
def get_snacks(drink: str) -> str:
    snacks = "cookies"
    if drink.lower() == "fruit juice":
        snacks = "banana"
    elif drink.lower() == "green tea":
        snacks = "nuts"
    elif drink.lower() == "hot chocolate":
        snacks = "bread"
    return snacks

#Creating a chain

agent = create_react_agent(
    model,
    [get_weather, get_temperature, get_drink, get_snacks],
    prompt=PromptTemplate.from_template(
        """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

\nQuestion: the input question you must answer
\nThought: you should always think about what to do
\nAction: the action to take, should be one of [{tool_names}]
\nAction Input: the input to the action
\nObservation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

\nThought: I now know the final answer
\nFinal Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}

"""
    ),
)

#Creating an agent

agent_executor = AgentExecutor(
    agent=agent,
    tools=[get_weather, get_temperature, get_drink, get_snacks],
    verbose=True,
    handle_parsing_errors=True,
)

# Creating a prompt wrapper as Runnable

augument_question = RunnableLambda(lambda x: {"input": "Please Help, " + x})

# Chaining prompt wrapper and agent

final_agent = augument_question | agent_executor

question = "I'm in chennai suggest a drink and snacks"

print(f"Question : {question}")

#Querying the agent

result = final_agent.invoke(question)

print(result)
