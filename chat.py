import os
import streamlit as st
import langchain
#from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from trubrics.integrations.streamlit import FeedbackCollector

langchain.verbose = True

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

collector = FeedbackCollector(
    component_name="LLM_Feedback",
    email=st.secrets["TRUBRICS_EMAIL"],
    password=st.secrets["TRUBRICS_PWD"],
)

# Initialize Streamlit app configuration
st.set_page_config(page_title="Gitcoin Grants Round 18")


index = './gg18-round/storage/faiss_index'
embeddings = OpenAIEmbeddings()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

vectors = FAISS.load_local(index, embeddings)

prompt_template = """We have provided context information below. 

{context}

If the answer is not available in the context information given above, respond: Sorry! I don't have an answer for this.
Given this information, please answer the following question in detail only using context information above. When sharing information about a project, always share its Project Details link in Explorer. 
Question: {question}"""

prompt_type = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": prompt_type}


chain = ConversationalRetrievalChain.from_llm(
	llm = ChatOpenAI(
		temperature=0.0,
		model_name='gpt-3.5-turbo'
		),
	retriever=vectors.as_retriever(),
	memory=memory,
	combine_docs_chain_kwargs=chain_type_kwargs,
	max_tokens_limit=3000
	)

def conversational_chat(query):
    
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    
    return result["answer"]


# Initialize chat history

st.session_state['history'] = []

if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_msg="Hi there! 👋 Let me know if you have any questions about the grantees in the GG18 Climate Round. I am still learning the ways around here, so always refer the project details on [Explorer](https://grants.gitcoin.co/#rounds) before you finalize your contribution. (Psst...I don't know anything about the Core Rounds other than Climate yet, but I will soon)"
    #st.chat_message("assistant").markdown(welcome_msg)
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("What would you like to know about grantees in the Climate Round"):
	#History

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Searching...brb...")        
        response = conversational_chat(prompt)
        message_placeholder.markdown(response)
    
    collector.st_feedback(feedback_type="thumbs", model="GG18", metadata={"Question": prompt, "Response": response},open_feedback_label = "[Optional] Provide any additional info:",)   

    #st.chat_message(str(feedback))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
