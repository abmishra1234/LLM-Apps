#Imports for the POC

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain



# Lets Start Creating Pieces here...
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
'''
#answer = llm.invoke("What are the top 05 programming language?")
#print(answer.content)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are in Expert of AI/ML development. \
     Answer my question in 03 parts : 1. Give the clear and concise answer, \
     2. Give the reason of your order of language\
     3. Put the final conclusion"),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# answer = chain.invoke({"input": "Top 05 language used in AI/ML development"})
# print(answer)

output_parser = CommaSeparatedListOutputParser()
msgs = output_parser.parse("hi, bye")
# print(msgs[0])
# print(msgs[1])

# Define a query
query = "List the top 5 programming languages in 2023."

# Function to simulate model's response
def simulate_model_response(query):
    # This is a placeholder for the actual model's API call
    # We're using a simulated string output as if coming from a language model
    return "1. Python, 2. JavaScript, 3. Java, 4. C#, 5. Go"

# Get the raw output from the language model
raw_output = simulate_model_response(query)

# Initialize the ListOutputParser
parser = CommaSeparatedListOutputParser()

# Parse the output into a list
parsed_list = parser.parse(raw_output)

# Print the results
print("Parsed List:", parsed_list)

'''



#loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
loader = WebBaseLoader("https://www.moneycontrol.com/")


docs = loader.load()
embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)


prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

q = "Why JSW Energy is in news?"

# Let's try to understand the below code with clarity
document_chain.invoke({
    "input": q,
    "context": [Document(page_content="This is financial site in india!!!")]
})

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": q})
print(response["answer"])

# This answer should be much more accurate!