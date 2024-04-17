import streamlit as st

def summarization_001(text):
    '''
        The first try to do the summerization of given
        text in some specified format
    '''
    from langchain_openai import ChatOpenAI
    from langchain.schema import(
        AIMessage,
        HumanMessage,
        SystemMessage
    )

    messages = [
        SystemMessage(content='You are an expert copywriter with expertize in summarizing documents'),
        HumanMessage(content=f'Please provide a short and concise summary of the following text:\n TEXT: {text}')
    ]

    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

    ntokens = llm.get_num_tokens(text)
    st.text(f'Number of token = {ntokens}')
    summary_output = llm.invoke(messages)
    #print(summary_output.content)
    return summary_output.content

def summarization_002(text, language):
    '''
        Summarizing Using Prompt Templates
        for specific language translation
    '''
    # here are some required imports
    from langchain_openai import ChatOpenAI
    from langchain import PromptTemplate
    from langchain.chains import LLMChain    

    template = '''
    Write concise and short summary of the following text :
    TEXT: `{text}`
    Translate the summary to {language}.
    '''

    prompt = PromptTemplate(
        input_variables=['text', 'language'],
        template=template
    )

    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    nTokens = llm.get_num_tokens(prompt.format(text=text, language='English'))
    st.text(f'Number of token = {nTokens}')

    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.invoke({'text': text, 'language':language})

    return summary['text']

def summarization_003():
    '''
       Summarizing using StuffDocumentChain
       meaning chain_type is 'stuff'
    
       There are two more option available here
       map-reduce type and refine type
    '''
    from langchain import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain.chains.summarize import load_summarize_chain
    from langchain.docstore.document import Document

    with open('transcript.txt', encoding='utf-8') as f:
        text = f.read()

    docs = [Document(page_content=text)]
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    template = '''Write a concise and short summary of the following text.
    TEXT: `{text}`. Consider your output in 03 sub section : Introduction, 
    Summary with Key points to be highlighted,Conclusion. 
    '''
    prompt = PromptTemplate(
        input_variables=['text'],
        template=template
    )

    chain = load_summarize_chain(
    llm,
    chain_type='stuff',
    prompt=prompt,
    verbose=False
    )
    output_summary = chain.invoke(docs)
    print(output_summary['output_text'])

def summarization_004():
    '''
    Title - Summarizing Large Documents Using map_reduce
    This method is for summarization using 
    chain_type : 'map-reduce'
    '''
    from langchain import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain.chains.summarize import load_summarize_chain
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    with open('test.txt', encoding='utf-8') as f:
        text = f.read()
    
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    ntoken = llm.get_num_tokens(text)

    # write the number of token to be passed LLM
    st.write(f'The token count = {ntoken}')

    # now time to split the document and create the chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.create_documents(text)
    st.write(f'the number of chunks = {len(chunks)}')

    # now it's time to create the appropriate chain to handle the 
    # summarization task for map_reduce chain_type
    chain = load_summarize_chain(
        llm,
        chain_type='map_reduce',
        verbose=False
    )

    output_summary = chain.invoke(chunks)
    return output_summary['output_text']

def summarization_005():
    '''
    map_reduce wich Custom Prompts
    '''
    from langchain import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain.chains.summarize import load_summarize_chain
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    with open('test.txt', encoding='utf-8') as f:
        text = f.read()
        
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

    llm.get_num_tokens(text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
    chunks = text_splitter.create_documents([text])

    st.write(len(chunks))

    map_prompt = '''
    Write a short and concise summary of the following:
    Text: `{text}`
    CONCISE SUMMARY:
    '''
    map_prompt_template = PromptTemplate(
        input_variables=['text'],
        template=map_prompt
    )

    combine_prompt = '''
    Write a concise summary of the following text that covers the key points.
    Add a title to the summary.
    Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the 
    topic FOLLOWED by BULLET POINTS if possible AND end the summary 
    with a CONCLUSION PHRASE.
    Text: `{text}`
    '''
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=['text'])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=False
    )
    output = summary_chain.invoke(chunks)
    # output_summary is a dict with 2 keys: 'input_documents' and 'output_text'
    # displaying the summary
    return (output['output_text'])

def main():
    '''
    This is my main entry point and our app starting from here.
    '''
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    # this is the first call of summaerization
    # Let's input your text
    text = st.text_input(label='Input your text for summerization')
    # Define the supported languages
    languages = {
        'English': 'en',
        'Spanish': 'es',
        'German': 'de',
        'Hindi' : 'hi'
        # Add more languages and their codes here
    }

    # Language selection widget with English as the default value
    selected_language_code = st.selectbox(
        'Select your language:',
        list(languages.keys()),
        index=list(languages.keys()).index('English')  # Default to English
    )

    if text:
        output_string = summarization_002(text, selected_language_code)
        st.text_area(label='Summarized Text Output', value=output_string)

    #   Testing the code for method 03 with better template
    #summarization_003()
    #result = summarization_004()
    #st.write(result)

# Entry method for streamlit application
if __name__ == "__main__":
    main()

