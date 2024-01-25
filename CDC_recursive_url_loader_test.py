#!/usr/bin/env python
# coding: utf-8

# In[168]:


get_ipython().system('pip install -q openai langchain beautifulsoup4 chroma chromadb tiktoken langchainhub crewai "unstructured[csv]"')


# In[65]:


from getpass import getpass
import os
os.environ['OPENAI_API_KEY'] = getpass()


# In[283]:


#Extract text from CDC website
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
import re
from html import unescape

import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import time

#Send to GPT4 for cleanup
def clean_text_with_gpt4(text):
    """
    This function takes a string of text and uses GPT-4 to clean it up using the OpenAI ChatCompletion API.
    It handles large texts by breaking them into smaller chunks.
    :param text: String containing the text to be cleaned.
    :return: Cleaned text as a string.
    Note: It's actually kind of inefficient and takes a really long time + money. Disabling it below.
    """
    cleaned_texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    chunks = text_splitter.split_documents(text)

    for chunk in chunks:
        try:
            print(chunk)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Assuming using the latest GPT-4 model
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": f"Please clean up the following text:\n\n{chunk}"}]
            )
            print(response)
            cleaned_texts.append(response.choices[0].message.content.strip())
            #print(cleaned_texts)
            time.sleep(1)  # Delay to respect rate limits
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    return ' '.join(cleaned_texts)

def clean_webcrawl_data(data):
    # Step 1a: Replace escape sequences like \n, \t, \r with a single space
    data = re.sub(r'\\[ntr]|\\x[0-9A-Fa-f]{2}', ' ', data)
    # Step 1b: Replace multiple spaces with a single space
    data = re.sub(r'\s+', ' ', data)
    # Step 2: Allow only ASCII characters (alphanumeric and basic punctuation)
    allowed_chars = r'[^\x00-\x7F]+'
    data = re.sub(allowed_chars, '', data)
    # Step 3: Unescape HTML entities
    data = unescape(data)
    # Step 4: Trim leading and trailing spaces
    data = data.strip()
    return data

def process_loaded_docs(documents):
    cleaned_documents = [clean_webcrawl_data(str(doc)) for doc in documents]  # Clean each document
    return cleaned_documents

file_name = "cleaned_texts.txt"
if os.path.exists(file_name):
    print(f"The file {file_name} already exists.")
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            cleaned_text = file.read()
            #print("File content:\n")
            #print(cleaned_text)
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    url = "https://www.cdc.gov"
    loader = RecursiveUrlLoader(
        url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs = loader.load()
    client = OpenAI()
    #cleaned_text = clean_text_with_gpt4(docs)
    cleaned_text = process_loaded_docs(docs)
    with open(file_name, 'a', encoding='utf-8') as file:
        for text in cleaned_text:
            file.write(str(text) + "\n")  # Adding two newlines as a separator between texts
    print(f"Cleaned texts saved to {file_name}")


# In[120]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}
        
def create_documents_from_text(text, chunk_size=1000):
    # Split the text into chunks of `chunk_size`
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    # Create a list of Document objects
    documents = [Document(chunk) for chunk in chunks]
    return documents

# Assuming `cleaned_text` is your cleaned text string
documents = create_documents_from_text(cleaned_text)
text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter2.split_documents(documents)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
CDC_retriever = vectorstore.as_retriever()


# In[232]:


import zipfile
from langchain_community.document_loaders import DirectoryLoader
import glob
import unstructured

# Step 1: Absolute path (replace with the actual absolute path)
absolute_path = './EpiHiper-Schema-master/'

# Step 2: Check if the directory exists
if not os.path.exists(absolute_path):
    print(f"Directory does not exist: {absolute_path}")
else:
    print(f"Directory exists: {absolute_path}")

    # Step 3: Manually list files
    file_paths = glob.glob(absolute_path + '/**/*.*', recursive=True)
    print(f"Manually found {len(files)} files:")
    for file_path in file_paths:
        print(file_path)
    file_contents_dict={}
    for file_path in file_paths:
        try:
            filename = os.path.basename(file_path)
            with open(file_path, 'r') as file_pt:
                content = file_pt.read()
            if filename in file_contents_dict:
                file_contents_dict[filename] += '\n' + content
            else:
                file_contents_dict[filename] = content
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

loader2 = DirectoryLoader('./EpiHiper-Schema-master/', glob="**/*.*", show_progress=True)
abm_codes = loader2.load()
# Debugging: Print the number of loaded files
print(f"Number of loaded files: {len(abm_codes)}")

#code_splits = split_by_length_with_overlap(abm_codes)
vectorstore = Chroma.from_documents(documents=abm_codes, embedding=OpenAIEmbeddings())
ABM_retriever = vectorstore.as_retriever()


# In[238]:


#for doc in abm_codes:
#    print(doc.metadata['source'])
disease_model_schema = file_contents_dict['diseaseModelSchema.json']
disease_model_rules = file_contents_dict['diseaseModelRules.json']


# In[250]:


from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
import textwrap

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=.2)
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.0) #"gpt-4-1106-preview", "gpt-4-0314"

query_planner = Agent(
    role="Simulation Planner",
    goal="Plan the steps needed to parameterize an agent-based simulation from existing knowledge",
    backstory=textwrap.dedent("""
        You are an expert at identifying modeling parameters from code base that implements 
        an agent-based model and listing the model choices, parameters, and json config files 
        whose values need to be determined. You will break down each model choice, parameter, 
        and json config file into sub-questions such that the answer to each sub-question will 
        inform the value to be used in the agent-based simulation.
        Accept the user-question and determine if it requires sub-questions to either the
        CDC website which provides an official source of recent infectious disease outbreaks
        or Wikipedia for information about a geographical location, country, infectious agent
        characteristics, transmission dynamics, infection states, or other epidemiological
        modeling efforts.
        Your final answer MUST be a description of sub-questions that explain the best model
        choices, model parameters, and config files for an agent-based modeling code base.
    """),
    verbose=True,
    allow_delegation=False,
    tools=[],  ###
    llm=llm,
)
step_planner = Agent(
    role="Plan Translator",
    goal="List plan steps in a python list of strings format",
    backstory=textwrap.dedent("""
        You know how to convert a plan such as this:
        
            Thought: Do I need to use a tool? No
            Final Answer: To effectively model the current flavivirus outbreak using an agent-based model, we need to answer a series of sub-questions that will inform the values and choices within the JSON configuration file. Here is an organized list of questions to be answered:

            Model Choice Sub-Questions:
            1. What is the specific flavivirus causing the outbreak (e.g., Zika, Dengue, West Nile)?
            2. What are the known vectors for transmission of this virus?
            3. What is the geographical scope of the model (e.g., a specific city, region, or country)?
            4. What is the time frame for the simulation (e.g., start and end dates)?

            Other extra text or thoughts.
            
        Into a format like this:
            steps = ["What is the specific flavivirus causing the outbreak (e.g., Zika, Dengue, West Nile)?","What are the known vectors for transmission of this virus?","What is the geographical scope of the model (e.g., a specific city, region, or country)?","What is the time frame for the simulation (e.g., start and end dates)?"]

        This is what you will do with the inputs provided.
    """),
    verbose=True,
    allow_delegation=False,
    tools=[],  ###
    llm=llm,
)


# In[177]:


#from langchain import hub
#prompt = hub.pull("hwchase17/self-ask-with-search")
#print(f'{prompt.format(agent_scratchpad = "AGENTSCRATCHPAD", input = "INPUT")}')

from langchain.tools.retriever import create_retriever_tool
from langchain.tools import DuckDuckGoSearchRun

CDC_retriever_tool = create_retriever_tool(
    CDC_retriever,
    "CDC_retriever_tool",
    """As an AI assistant you provide answers based on the given context, ensuring accuracy and briefness. 

        You always follow these guidelines:

        -If the answer isn't available within the context, state that fact
        -Otherwise, answer to your best capability, refering to source of documents provided
        -Only use examples if explicitly requested
        -Do not introduce examples outside of the context
        -Do not answer if context is absent
        -Limit responses to three or four sentences for clarity and conciseness
        
        Search for data related to outbreaks. For questions about outbreaks, use this tool to return 
        relevant data for answering questions about outbreaks""",
)

ABM_retriever_tool = create_retriever_tool(
    ABM_retriever,
    "ABM_retriever_tool",
    """As an AI assistant you provide answers based on the given context, ensuring accuracy and briefness. 

        You always follow these guidelines:

        -If the answer isn't available within the context, state that fact
        -Otherwise, answer to your best capability, refering to source of documents provided
        -Only use examples if explicitly requested
        -Do not introduce examples outside of the context
        -Do not answer if context is absent
        -Limit responses to three or four sentences for clarity and conciseness
        
        Search for data related to outbreaks. For questions about outbreaks, use this tool to return 
        relevant data for answering questions about outbreaks""",
)

web_search = DuckDuckGoSearchRun()

query_executor = Agent(
    role="""
        Agent Role: Information Searcher

        Primary Objectives:
        1. Utilize the CDC_retriever_tool to gather current outbreak information. This includes statistics, affected regions, and latest guidelines related to the outbreak.

        2. Employ the ABM_retriever_tool to access the codebase for agent-based modeling simulations. Extract relevant parameters and settings that are crucial for understanding the dynamics of the disease spread in the simulations.

        3. Conduct thorough internet searches using the web_search tool. Focus on disease-specific information such as modes of transmission, vectors involved, the role of asymptomatic infectious carriers, and insights from past modeling efforts.

        Key Responsibilities:
        - Ensure accurate and up-to-date information is retrieved from each tool.
        - Synthesize information from diverse sources to provide a comprehensive understanding of the disease and its impact.
        - Adhere to the principles of clarity and conciseness in reporting findings.
        """,
    goal="Information Searcher",
    backstory=textwrap.dedent("""
        Accept list of sub-questions from the query_planner agent and perform
        the necessary searches to answer the questions.
        Perform the tasks in the order given and report the result out.
        Your final answer MUST be a correct response to the original user-query.
    """),
    verbose=True,
    llm=llm,
    # tools=[SqlTools.do_sql_query, RagTools.do_rag_query],
    tools=[CDC_retriever_tool, ABM_retriever_tool, web_search],
    allow_delegation=True,
)

param_executor = Agent(
    role="""
        Agent Role: Model Parameterizer

        Primary Objectives:
        1. Identify the json schema in the context and assess the information needed to assign values to all json fields.

        2. Examine the context for additional information from prior prompts and searches and use those to assign value to each json field.

        3. Produce a json file in the same format as the json schema with the field values filled in according to the information provided.

        Key Responsibilities:
        - Make use of information provided
        - Synthesize information from diverse sources to provide a comprehensive understanding of the disease and its impact.
        - Adhere to the json format in your final output.
        """,
    goal="Parameterize an agent-based model",
    backstory=textwrap.dedent("""
        Take in a json schema file and output an updated json file with values filled in based on information provided
    """),
    verbose=True,
    llm=llm,
    # tools=[SqlTools.do_sql_query, RagTools.do_rag_query],
    tools=[CDC_retriever_tool, web_search],
    allow_delegation=True,
)

critic = Agent(
    role="Evaluate Answer",
    goal="Provide feedback on prior responses to user query",
    backstory=textwrap.dedent("""
        You are an expert at understanding, correcting, and producing json formatted strings.
        You know how to list the model choices and parameters in a json schema file format and
        how to convert text about model choices and parameters into a json schema, following
        any examples provided.
        You will provide critical feedback and improve the final output using that feedback.
        Your final answer MUST be a json schema file for an agent-based modeling code base.
    """),
    verbose=True,
    allow_delegation=False,
    tools=[],  ###
    llm=llm,
)


# In[289]:


param_executor_old = Agent(
    role="""
        Agent Role: Model Parameterizer

        Primary Objectives:
        1. Identify the json schema in the context and assess the information needed to assign values to all json fields.

        2. Examine the context for additional information from prior prompts and searches and use those to assign value to each json field.

        3. If there is any missing information that is needed, ask very specific questions and use CDC_retriever_tool or web_search to find the needed information.

        4. Produce a json file in the same format as the json schema with the field values filled in according to the information provided.

        Key Responsibilities:
        - Make use of information provided
        - Synthesize information from diverse sources to provide a comprehensive understanding of the disease and its impact.
        - Adhere to the json format in your final output.
        """,
    goal="Parameterize an agent-based model",
    backstory=textwrap.dedent("""
        Take in a json schema file and output an updated json file with values filled in based on information provided
    """),
    verbose=True,
    llm=llm,
    # tools=[SqlTools.do_sql_query, RagTools.do_rag_query],
    tools=[CDC_retriever_tool, web_search],
    allow_delegation=True,
)

json_validator_old = Agent(
    role="JSON format validator",
    goal="Make sure the final answer is in the appropriate json format",
    backstory=textwrap.dedent("""
        You are an expert at understanding, correcting, and producing json formatted strings.
        You know how to list the model choices and parameters in a json schema file format and
        how to convert text about model choices and parameters into a json schema, following
        any examples provided. 
        Your final answer MUST be a json schema file for an agent-based modeling code base.
    """),
    verbose=True,
    allow_delegation=False,
    tools=[],  ###
    llm=llm,
)

task1 = Task(
    description=textwrap.dedent(f"""
        Your task is to go through a json file in the agent-based model code base and set the 
        values for each field that needs to be parameterized in the below schema:
            {disease_model_schema}
        Use the CDC_retriever_tool data about current outbreaks and web_search find general 
        disease information and characteristics and produce a json output with the values 
        needed to carry out the below user query: 
            {user_query}
        The final output should be a json formatted string.
    """),
    agent=param_executor
)

task2 = Task(
    description=textwrap.dedent(f"""
        You will recieve a set of queries and information from the previous task. Your task is
        make sure the information gathered conform to the json schema format provided.
        Your final answer must be be a json formatted string.
    """),
    agent=json_validator
)

task2b = Task(
    description=textwrap.dedent(f"""
        Your task is to go through a json file in the agent-based model code base and set the 
        values for each field that needs to be parameterized in the below schema:
            {disease_model_schema}
        Use the CDC_retriever_tool data about current outbreaks and web_search find general 
        disease information and characteristics and produce a json output with the values 
        needed to carry out the below user query: 
            {user_query}
        The final output should be a json formatted string.
    """),
    agent=param_executor
)

#Example that got me to JSON param file #1
crew = Crew(
    agents=[param_executor_old,json_validator_old],
    tasks=[task1,task2],
    verbose=2,  # print what tasks are being worked on, can set it to 1 or 2
    process=Process.sequential,
)

result = crew.kickoff()

print("######################")
print(result)


# In[293]:


user_query = "Model the current flavivirus outbreak using an agent based model"

planning_task = Task(
    description=textwrap.dedent(f"""
        Your task is to plan out the necessary steps in order to fulfull the user task and 
        create smaller prompts for other co-workers to follow. The result of this task should
        be an organized list of questions to be answered in order to fulfill the user request.
            {user_query}
        Use the following json template that comes from an agent-based model code base to help
        you decide what subtasks need to be completed to fulfull the user request:
            {disease_model_schema}
    """),
    agent=query_planner
)

list_steps_task = Task(
    description=textwrap.dedent(f"""
        You will recieve a set of subquestions from the previous task. Your task is to format
        them as a list of steps saved as a python list of strings.
    """),
    agent=step_planner
)

critic_task = Task(
    description=textwrap.dedent(f"""
        You will recieve a set of queries and information from the previous task. You will:
        1. Examine the JSON output and critique it based on information provided, decide if the 
        numbers are numerically reasonable in light of the provided data.
        2. If absolute numbers are not available from in the provided data, then you will 
        examine the relative values of the field values to see if they are related appropriately,
        i.e., greater than, less than, or approximately equal. 
        3. You will list field values that should be altered from points 1 and 2, their original
        value, and suggest a new value.
        4. You will incorporate those values into the full JSON schema format and check for proper 
        JSON formatting and make any corrections needed.
        5. Your final answer MUT be be a json formatted string.
    """),
    agent=json_validator
)


# In[294]:


crew = Crew(
    agents=[query_planner,step_planner],
    tasks=[planning_task,list_steps_task],
    verbose=2,  # print what tasks are being worked on, can set it to 1 or 2
    process=Process.sequential,
)

result = crew.kickoff()

print("######################")
print(result)

#get output of list_steps_task into a python list
test = result.replace('\n','').replace("`",'').replace("  ",'')
import ast
test2= ast.literal_eval(test)

#for q in test2:
#    print(q)

answers = []
for q in test2:
    subquestion_task = Task(
        description=textwrap.dedent(f"""
            Your task is to find the information needed to answer the question:
                {q}
            In the context of the original user request:
                {user_query}
        """),
        agent=query_executor
    )
    crew = Crew(
        agents=[query_executor],
        tasks=[subquestion_task],
        verbose=2,  # print what tasks are being worked on, can set it to 1 or 2
        process=Process.sequential,
    )

    answer = crew.kickoff()
    answers.append(answer)

print("######################")
print(answers)

answers_dump = '\n'.join(answers)

compile_params_task = Task(
    description=textwrap.dedent(f"""
        USEFUL INFORMATION:
            {answers_dump}
        Your task is to go through a json file in the agent-based model code base and set the 
        values for each field that needs to be parameterized in the below schema:
            {disease_model_schema}
        Use the USEFUL INFORMATION to produce a json output with the values 
        needed to carry out the below user query: 
            {user_query}
        The final output should be a json formatted string.
    """),
    agent=param_executor
)

crew = Crew(
    agents=[param_executor,critic],
    tasks=[compile_params_task,critic_task],
    verbose=2,  # print what tasks are being worked on, can set it to 1 or 2
    process=Process.sequential,
)

result2 = crew.kickoff()

print("######################")
print(result2)



# In[292]:


param_executor2 = Agent(
    role="""
        Agent Role: Model Parameterizer

        Primary Objectives:
        1. Identify the json schema in the context and assess the information needed to assign values to all json fields.

        2. Examine the context for additional information from prior prompts and searches and use those to assign value to each json field.

        3. Produce a json file in the same format as the json schema with the field values filled in according to the information provided.

        Key Responsibilities:
        - Make use of information provided
        - Synthesize information from diverse sources to provide a comprehensive understanding of the disease and its impact.
        - Adhere to the json format in your final output.
        """,
    goal="Parameterize an agent-based model",
    backstory=textwrap.dedent("""
        Take in a json schema file and output an updated json file with values filled in based on information provided
    """),
    verbose=True,
    llm=llm,
    # tools=[SqlTools.do_sql_query, RagTools.do_rag_query],
    tools=[CDC_retriever_tool, web_search],
    allow_delegation=True,
)

task5 = Task(
    description=textwrap.dedent(f"""
        USEFUL INFORMATION:
            {answers_dump}
        Your task is to go through a json file in the agent-based model code base and set the 
        values for each field that needs to be parameterized in the below schema:
            {disease_model_schema}
        Produce a modified json schema with the values needed to carry out the below user request: 
            {user_query}
        The final output should be a json formatted schema that has been updated to different 
        field values than the initial example to incorporate as much information as possible
        about the disease mentioned in the user request. When uncertain, try to use estimate
        values.
    """),
    agent=param_executor
)

crew = Crew(
    agents=[param_executor2,critic],
    tasks=[task5,task2b],
    verbose=2,  # print what tasks are being worked on, can set it to 1 or 2
    process=Process.sequential,
)

result2 = crew.kickoff()

print("######################")
print(result2)


# In[ ]:




