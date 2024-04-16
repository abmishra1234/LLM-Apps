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


# Entry method for streamlit application
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    # this is the first call of summaerization

    # Let's input your text
    text = st.text_input(label='Input your text for summerization')
    if text:
        output_string = summarization_002(text, 'Hindi')
        st.text_area(label='Summarized Text Output', value=output_string)



