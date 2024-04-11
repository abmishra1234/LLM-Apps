import streamlit as st
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import os


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# Chunking Data
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# Create Embedding
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

# Asking and Getting Answer
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    return answer['result'] # return only answer and not query

# Calculate the Cost
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image('banner.png')
    st.subheader('GenAI LLM Powered Assistant of your private document 🤖')
    
    with st.sidebar:
        st.subheader('Document Upload Panel')
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk Size:', min_value=100, max_value=2048, value=512)
        k = st.number_input('k (number of chunks passed to LLM from chunk database)', min_value=1, max_value=10, value=3)
        add_data = st.button('Add Data')

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

                vector_store = create_embeddings(chunks)
                # now save this vector_store into user's session
                st.session_state.vs = vector_store

                st.success('File uploaded, chunked and embedded successfully!')

    # Let's start main part of your RAG Application
    q = st.text_input('Ask any question about the content of your file:')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'value of k : {k}')

            # now let's call the llm for answer
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area('Assistant Answer: ', value=answer)
            st.divider()
            # Lets start working on making the Chat history below part of your main window
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Question: {q} \nAnswer: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)

    # End of Application Logic for now...





