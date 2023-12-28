from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool, YouTubeSearchTool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector
import streamlit as st
from langchain.vectorstores import Cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from datasets import load_dataset
from utils import write_message

# tag::setup[]
# Page Config
st.set_page_config("Movie Expert", page_icon=":movie_camera:")
# end::setup[]

SYSTEM_MESSAGE = """
You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to movies, actors or directors.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.
"""

# Load environment variables from .env file

youtube = YouTubeSearchTool()
embedding_provider = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
chat_llm = ChatOpenAI(openai_api_key = st.secrets["OPENAI_API_KEY"])

#####

'''
    Cassandra AstraDB part
'''
keyspace_name = "qa_docs"
table_name = "qa_table"



cloud_config= {
  'secure_connect_bundle': 'secure-connect-db-competition.zip'
}
auth_provider = PlainTextAuthProvider(st.secrets["ASTRADB_ID"], st.secrets["ASTRADB_SECRET"])
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()


astra_vector_store = Cassandra(
    embedding=embedding_provider,
    session=session,
    keyspace=keyspace_name,
    table_name=table_name,
)

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

def initialize_dataset():
    NUM_PLOTS = 100

    print("Loading data from huggingface ... ", end="")
    movie_plot_dataset = load_dataset("vishnupriyavr/wiki-movie-plots-with-summaries", split="train[:10]")

    def concatenate_columns(example):
        return {'concatenated_text': "Movie name: "+ example['Title'] + ". Movie Plot: " + example['Plot']}

    # Apply the function to the dataset
    movie_plot_dataset_concat = movie_plot_dataset.map(concatenate_columns)

    # Verify the result
    #print(movie_plot_dataset_concat['concatenated_text'])
    concatenated_texts = [example['concatenated_text'] for example in movie_plot_dataset_concat]
    print(concatenated_texts)

    print("Done.")

    print("\nGenerating embeddings and storing Plots in AstraDB ... ", end="")
    astra_vector_store.add_texts(concatenated_texts)

    print("Inserted %i headlines." % len(concatenated_texts))

# tag::session[]
# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the MovieExpert Chatbot!  How can I help you?"},
    ]

    initialize_dataset()

def run_retriever_astradb(query):
    print(query)
    answer = astra_vector_index.query(query, llm=chat_llm).strip()
    print(answer)
    return answer


prompt = PromptTemplate(
    template="""
    You are a movie expert. You find movies from a genre or plot.

    ChatHistory:{chat_history}
    Question:{input}
    """,
    input_variables=["chat_history", "input"]
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
        func=run_retriever_astradb,
        return_direct=False
    )

         ]

agent = initialize_agent(
    tools, chat_llm, memory=memory,
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


# end::session[]

# tag::submit[]
# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        
        response = generate_response(message)
        write_message('assistant', response)
# end::submit[]


# tag::chat[]
with st.container():
    # Display messages in Session State
    for message in st.session_state.messages:
        write_message(message['role'], message['content'], save=False)

    # Handle any user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        write_message('user', prompt)

        # Generate a response
        handle_submit(prompt)
# end::chat[]
