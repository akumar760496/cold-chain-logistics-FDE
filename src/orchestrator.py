import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated , TypedDict


from langchain_core.messages import BaseMessage , SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


# setup path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0,str(project_root))

from src.agent_tools import query_telemetry_db, fetch_corridor_conditions, search_compliance_sop

load_dotenv(project_root / ".env")

class AgentState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]


# Factory initialization of LLM

AGENT_LLM_SETTING = os.getenv("Agent_llm", "OLLAMA").strip().upper()


if AGENT_LLM_SETTING == "OPENAI":
    print("Initializing Agent with OpenAI")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

elif AGENT_LLM_SETTING == "DEEPSEEK":
    print("Initializing llm with deepseek")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=0,
        openai_api_key = os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        max_tokens = 2048,
    )

else:
    print("Using fallback ollama model qwen2.5:7b")
    from langchain_community.chat_models import ChatOllama
    llm = ChatOllama(model="qwen2.5b", temperature = 0, num_predict = 1024)

fde_tools = [query_telemetry_db, search_compliance_sop, fetch_corridor_conditions]
llm_with_tools = llm.bind_tools(fde_tools)

# Langgraph architecture assembly

def reasoning_node(state:AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response]
    }

print("Compiling Langgraph FDE Orchestrator")
graph_builder = StateGraph(AgentState)
graph_builder.add_node("reasoner",reasoning_node)
graph_builder.add_node("tools",ToolNode(fde_tools))

graph_builder.add_edge(START, "reasoner")
graph_builder.add_conditional_edges("reasoner", tools_condition)
graph_builder.add_edge("tools", "reasoner")

fde_agent = graph_builder.compile(checkpointer= MemorySaver())


# Chat loop testing panel 
if __name__ == "__main__":
    print("FDE Supply chain Orchestrator State Mahcine ")
    print(f"-- Configured execution : [LLM : {AGENT_LLM_SETTING}] -> [Embeddings : {os.getenv("Embeddings_model",'LOCAL')}]")

    prompt_path = project_root / "src" / "prompts" / "system_prompt.txt"

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_instructions = f.read()
    except FileNotFoundError:
        print(f"Error : could not find prompt path - {prompt_path}")
        system_instructions = "You are a helpful AI assistant."

    system_prompt = SystemMessage(content=system_instructions)

    thread_config = {"configurable": {"thread_id":"production_test_1"}}
    fde_agent.invoke({"messages": [system_prompt]}, config=thread_config)

    while True:
        user_input = input("\\ Dispatcher > ")
        if user_input.lower() in ["exit", "quit"]:
            break;

        events = fde_agent.stream({
            "messages": [("user", user_input)]
        },
        config= thread_config , stream_mode= "updates"
        )

        for event in events:
            for node_name , node_state in event.items():
                if node_name == "tools":
                    print("[system] Retrieveing data elements via ToolNode")
                elif node_name == "reasoner":
                    latest_msg = node_state["messages"][-1]
                    if latest_msg.content:
                        print(f"\n FDE Agent : \n {latest_msg.content}")


