from langchain.agents import AgentType, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from llm import llm
from langchain.tools import Tool, YouTubeSearchTool
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory

SYSTEM_MESSAGE = """
You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to movies, actors or directors.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.
"""

tools = []

youtube = YouTubeSearchTool()

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)

memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", return_messages=True )

chat_chain = LLMChain(llm=chat_llm, memory=memory, prompt=prompt, verbose=True)

tools = [
    Tool.from_function(
        name="ChatOpenAI",
        description="For when you need to chat about movies, genres or plots. The question will be a string. Return a string.",
        func = chat_chain.run,
        return_direct=True
    ),
    Tool.from_function(
        name="YouTubeSearchTool",
        description= "For when you need a link to a movie trailer. The question will be a string. Return a link to a YouTube video.",
        func = youtube.run,
        return_direct=True
    ),
    Tool.from_function(
        name="PlotRetrieval",
        description="For when you need to compare a plot to a movie. The question will be a string. Return a string.",
        func=run_retriever,
        return_direct=True
    )

         ]

agent = initialize_agent(
    tools, llm, memory=memory,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    max_iterations=3,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": SYSTEM_MESSAGE}
)


def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = agent(prompt)

    return response['output']