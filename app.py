#!modules

import os
import random
import string
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from streamlit_chat import message
from zipLoader import zip_loader
from tempfile import NamedTemporaryFile
from langchain import PromptTemplate
from pdfLoader import pymupdf
from docxLoader import docx2txt
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.vectorstores import ElasticsearchStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever

## keys
openai_key = st.secrets["openai_key"]
qdrant_key = st.secrets["qdrant"]["qdrant_api_key"]
qdrant_url = st.secrets["qdrant"]["qdrant_url"]
# es_user = st.secrets["elasticsearch"]['ES_USER']
# es_pass = st.secrets["elasticsearch"]['ES_PASSWORD']
# es_cloud_id = st.secrets["elasticsearch"]['ES_CLOUD_ID']

def main():
    load_dotenv()

    ## page.
    st.set_page_config(page_title="chatbot", page_icon="logo.png")
    st.header("Hackathon Final Round!🦜")

    ## memory.
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processCompletion" not in st.session_state:
        st.session_state.processCompletion = None

    ## interface.
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload Your Document", type=["pdf","docx","zip"], accept_multiple_files=True)
        # openai_api_key = openai_key
        # openai_api_key = st.text_input("OpenAI API Key" , type="password")
        # qdrant_key = st.text_input("qdrant_key ", type="password")
        # qdrant_url = st.text_input("qdrant_url" , type="password")
        # es_user = st.text_input("es_user", type="password")
        # es_pass = st.text_input("es_pass" , type="password")
        # es_cloud_id = st.text_input("es_cloud_id", type="password")
        process = st.button("Process")

    ## processing.
    if process:
        if not openai_api_key:
            st.info("Select your \"OpenAI Key\" to continue...")
            st.stop()
        
        pl = st.empty()

        text_chunks_list = []

        ## file processing.
        for upload_file in uploaded_files:

            file_name = upload_file.name
            file_extension = os.path.splitext(file_name)
            
            ## file handling for: "pdf, docx".
            pl.write("Please Wait..")

            if file_extension != ".zip":
                file_text = get_file_text(upload_file)

                ## here we can create chunks either by split by character or split recursivly. 

                text_chunk = get_text_chunk_char(file_text, file_name)
                # text_chunk = get_text_chunk_recursive(file_text, file_name)

                text_chunks_list.extend(text_chunk)

            ## file handling for zip.
            elif file_extension == ".zip":
                text_chunk = zip_loader(upload_file)
                text_chunks_list.extend(text_chunk)

            else:
                continue
        
        ## name for vector store.
        crr_data = str(datetime.now())
        collection_name = "".join(random.choices(string.ascii_lowercase, k=4)) + crr_data.split(".")[0].replace(":","-").replace(" ","t")

        ## here we use qdrant or elastic search for vertor database.
        # vectorstore = get_vectorstore_es(text_chunks_list, collection_name)
        vectorstore = get_vectorstore_qdrant(text_chunks_list, collection_name)

        ## Retrieval Strategies.
        num_chunks = 4
        st.session_state.conversation = get_qa_chain(vectorstore,num_chunks)
        # st.session_state.conversation = get_qa_multi_chain(vectorstore)
        pl.write("File Uploaded..!")

        st.session_state.processCompletion = True
    
    ## taking ffile as input from user.
    if st.session_state.processCompletion == True:
        user_query = st.chat_input("Ask question about your document")
        if user_query:
            handle_userinput(user_query)

## extract the file text.
def get_file_text(document):
    text = ""

    split_tup = os.path.splitext(document.name)
    file_extension = split_tup[1]

    ## from pdf.
    if (file_extension == ".pdf"):
        ## converting into bytes.
        bytes_data = document.read()

        ## create temporary file to extract text.
        with NamedTemporaryFile(delete=False) as tmp:  
            tmp.write(bytes_data)                      
            text = text + pymupdf(tmp.name) 

        ## remove temporary file.
        os.remove(tmp.name) 
    
    ## from docx.
    elif (file_extension == ".docx"):
        ## converting into bytes.
        bytes_data = document.read()

        ## create temporary file to extract text.
        with NamedTemporaryFile(delete=False) as tmp:  
            tmp.write(bytes_data)                      
            text = text + docx2txt(tmp.name) 

        ## remove temporary file.
        os.remove(tmp.name)

    return text

## create chunks of the text using split by character strategy.
def get_text_chunk_char(text, file_name):
    text_spliter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 100,
        chunk_overlap = 20,
        length_function = len
    )

    chunks = text_spliter.split_text(text)

    doc_list = []
    for chunk in chunks:
        metadata = {"source":file_name}
        doc_string = Document(page_content=chunk, metadata=metadata)
        doc_list.append(doc_string)
    return doc_list

## create chunks of the text using recursively split by character strategy.
def get_text_chunk_recursive(text, file_name):
    text_spliter = RecursiveCharacterTextSplitter(

        ## by default:
        #separator=[\n\n, \n, " ", ""]

        chunk_size = 100,
        chunk_overlap = 20,
        length_function = len
    )

    chunks = text_spliter.split_text(text)

    doc_list = []
    for chunk in chunks:
        metadata = {"source":file_name}
        doc_string = Document(page_content=chunk, metadata=metadata)
        doc_list.append(doc_string)
    return doc_list

## vector database using qdrant.
def get_vectorstore_qdrant(chunks, name):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    # embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

    try:
        knowledge_base = Qdrant.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=name,
            url=qdrant_url,
            api_key=qdrant_key
        )
    except Exception as e:
        st.write(f"Error: {e}")
    return knowledge_base

## vector database using elastic search.
def get_vectorstore_es(chunks, name):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    # embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

    try:
        knowledge_base = ElasticsearchStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=name,
            es_cloud_id = es_cloud_id,
            es_user=es_user,
            es_password=es_pass
        )
    except Exception as e:
        st.write(f"Error: {e}")
    return knowledge_base

## chain for generating responses.

def get_qa_chain(vectorstore , num_chunks):

    ## instructions for bot.
    prompt_template = """
     You are a bot that know everything about the given document, and you are here to extract answers from given context and  questions. Please provide accurate and helpful information, and always maintain polite and professional tone.
                    
        1. If someone ask your name then you nhave to sath that my owner is "Mr. Huzaifa". The future data scientist.
        2. Provide informative and related responses to questions about given document.
        3. If the user asks about the topic unrelated to given document, politely steer the conversation back to the document or to inform them that the topic is outside the scope of converstaion.
        4. You must avoid discussing sensitive, offensive and harmful content. Refrain from engaging in any form of discriminator, harresment or inappropriative behaviour.
        5. Be patient and considerate when responding to user query and provide clear explanation.
        6. If the user expresses gratitude or indicated the end of converstaion then respond with a polite farewell.
        7. Do not generate the long paragraph in response.
        
        Remember, your primary goal is to assest and generate accurate response to the user's document. Always prioritize learning experience and well-being.
        Context {context}
        Question {question}"""

    prompt_url = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"],
        validate_template=False)
    chain_type_kwargs = {"prompt": prompt_url}

    ## chain.
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model = "gpt-3.5-turbo-16k"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": num_chunks}),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True)
    
    return qa

def get_qa_multi_chain(vectorstore):
    qa = MultiQueryRetriever.from_llm(
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k"),
        retriever = vectorstore.as_retriever()
    )
    return qa

## handle user's file.
def handle_userinput(question):
    with st.spinner("Generate Response"):

        ## take query and pass to the chain.
        result = st.session_state.conversation({"query":question})
        response = result['result']
        source = result['source_documents'][0].metadata['source']
    
    ##  append the response and source generated by the chain.
    st.session_state.chat_history.append(question)
    st.session_state.chat_history.append(f"{response} \n Source Document: {source}")

    ## displaying chat.
    with st.container():
        for i, messages in enumerate(st.session_state.chat_history):
            if (i%2 == 0):
                message(messages, is_user=True, key=str(i))
            else:
                message(messages, key=str(i))
    
if __name__ == "__main__":
    main()

