import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
# for handling the Logging in project
import logging


# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    logging.info(f'Attempting to load document: {file}')
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        logging.info(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        logging.info(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        logging.info('Document format is not supported!')
        return None

    data = loader.load()
    logging.info('Document loaded successfully.')    
    return data

# Chunking Data
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    logging.info(f'chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}')
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    logging.info(f'chunk_data method complete successfully!!!')
    return chunks

# Create Embedding
def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    logging.info(f'create_embeddings_chroma method called with persist_directory=: {persist_directory}')

    # Instantiate an embedding model from OpenAI (smaller version for efficiency)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    # Create a Chroma vector store using the provided text chunks and embedding model,
    # configuring it to save data to the specified directory
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)

    logging.info(f'create_embeddings_chroma method completed successfully!!!')
    return vector_store  # Return the created vector store

# Load Embedding from the chroma_db disk location
def load_embeddings_chroma(persist_directory='./chroma_db'):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    # Instantiate the same embedding model used during creation
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    # Load a Chroma vector store from the specified directory, using the provided embedding function
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    return vector_store  # Return the loaded vector store

# Asking and Getting Answer
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    logging.info(f'The method ask_and_get_answer called with query: {q}, and the value of k= {k}')
    # Let's integerate the Prompt template here for your response refinement
    # TBD - 18042024 - Task 01 - Integerate Prompt template
    # TBD - 18042024 - How you will implement response in user preferred language
    #  update your UI and selected language. Please note default language must be selected
    # English, until other language is not supported
    # This will help you to achieve multi language support in response
    
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    logging.info(f'The method ask_and_get_answer completed successfully!!!')
    return answer['result'] # return only answer and not query

# Calculate the Cost
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    logging.info(f'The cost of embedding is {total_tokens / 1000 * 0.00002}, and the model used text-embedding-3-small')
    return total_tokens, total_tokens / 1000 * 0.00002

# The method for clearing the history
def history_clear():
    if 'history'in st.session_state:
        del st.session_state['history']
    logging.info('history cleaning completed successfully!!!')

# Load existing chat history from a file
def load_chat_history():
    logging.info('reading of chat_history.txt initiated!!!')
    try:
        with open("chat_history.txt", "r") as file:
            chat_history = file.read()
    except FileNotFoundError:
        logging.info('chat_history.txt file not found')
        chat_history = ""
    return chat_history

# Save chat history to a file
def save_chat_history(chat_history):
    logging.info(f'Called save_chat_history with {chat_history}')
    with open("chat_history.txt", "w") as file:
        file.write(chat_history)

# Define a function to handle loading history into session state
def handle_load_history():
    logging.info('handle_load_history called!!!')
    st.session_state.history = load_chat_history()
    st.rerun()

def build_side_panel(st):
    '''
    This code block is used for prepairing the side block panel 
    which is hidden for most of the use except admin
    As of now , it is viible to everyone but later it will be done according to
    the concept we thought.
    '''
    logging.info('build_side_panel method is called !!!')
    with st.sidebar:
        st.subheader('Document Upload Panel')
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk Size:', min_value=100, max_value=2048, value=512, 
                                        on_change=history_clear)
        k = st.number_input('Top k (number of chunks found based on sementic search)', 
                            min_value=1, max_value=10, value=3, on_change=history_clear)

        # Create two columns
        col_add_data, col_upload_embedding = st.columns(2)

        # Place a button in each column
        with col_add_data:
            add_data = st.button('Add Data', on_click=history_clear)

        with col_upload_embedding:
            upload_embedding = st.button('Upload the Existing Embedding')

        # Upload block
        if uploaded_file and add_data:
            with st.spinner('Document reading, chunking and embedding in progress...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                # write the chunk_size and chunks len
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                # calculate the embedding cost 
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding Cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings_chroma(chunks)
                # now save this vector_store into user's session
                st.session_state.vs = vector_store

                st.success('File uploaded, chunked and embedded successfully!')
        
        # Case when you don't want to chunk and embed to the same document again 
        if upload_embedding:
            with st.spinner('Loading of embedding in progress...'):
                vector_store = load_embeddings_chroma()
                # now save this vector_store into user's session
                st.session_state.vs = vector_store
                st.success('Loading of embedding Completed!')
        logging.info('build_side_panel method is Successfully completed !!!')

def build_main_window(st):
    banner_img = 'banner.png'
    st.image(banner_img)
    logging.info(f'Adding the background image = {banner_img}')
    st.subheader('GenAI LLM Powered Assistant of your private document 🤖')

    # Let's start main part of your RAG Application
    q = st.text_input('Ask any question about the content of your file:')
    answer = ''
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs

            # now let's call the llm for answer
            answer = ask_and_get_answer(vector_store, q)
            st.text_area('Assistant Answer: ', value=answer)

    st.divider()
    # Lets start working on making the Chat history below part of your main window
    if 'history' not in st.session_state:
        st.session_state.history = ''
    if q and q.strip():
        value = f'Question:\n{q} \n\nAnswer:\n{answer}'
        st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
        h = st.session_state.history

    # Create two columns
    col_load_history, col_save_history = st.columns(2)

    # Place a button in each column
    with col_load_history:
        if st.button('Load History'):
            handle_load_history()            

    with col_save_history:
        if st.button('Save History', on_click=save_chat_history(st.session_state.history)):
            # Code to save history
            st.write('History saved!')

    st.text_area(label='Chat History', value=st.session_state.history, key='history', height=400)

def main():
    import os
    from dotenv import load_dotenv, find_dotenv
    logging.info('__main__ method is called!!!')
    load_dotenv(find_dotenv(), override=True)

    build_side_panel(st)
    build_main_window(st)

    # End of Application Logic for now...

# Entry method for streamlit application
if __name__ == "__main__":
    main()