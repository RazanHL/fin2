#pip install google-search-results

import os
import json
import pandas as pd
import markdown
import time
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import groq
from langchain_groq import ChatGroq
import streamlit as st
from streamlit_chat import message
from langchain.tools import Tool
# from langchain_community.tools import DuckDuckGoSearchRun, BaseTool, QuerySQLDataBaseTool
import agents_tools
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import SystemMessagePromptTemplate
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
# from IPython.display import Markdown, display
# display(Markdown(output_text))


# file = open("/content/companies_names_from_edgar.json", "r")
# content = json.load(file)
# companies_names = pd.DataFrame(content).T
# companies_names.to_csv("companies_names.csv", index=False)

# load_dotenv()
os.environ["HUGGINGFACE_ACCESS_TOKEN"] ="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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] ="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" #os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ["SERPAPI_API_KEY"] = "61c1b0f3bafa9fa6a0c7375517eb3717c45cdd0dcfe67e6bfa1380aaee329ccc"

old = "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"
new = "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"
os.environ['GROQ_API_KEY'] = old
groq_api_key = old

llm=ChatGroq(groq_api_key=groq_api_key,
                model_name="llama3-70b-8192",
                temperature=0.2,
                max_retries=2,
            )

st.header("AI Financial Assistance")
# serpapi_key = 'bb1aa0326599cfdec4b017f65bc39a4618f13ba9947191ac2971d4cc5a72466e'

tools=[
    Tool(
        name="get stock data",
        func=agents_tools.get_recent_data,
        description="Use only when you are asked to evaluate or analyze a stock, don't use if you don't have \
                any company name. This will output historic share price data. You should input the stock ticker \
                to it "
    ),
    Tool(
        name="company overview",
        func=agents_tools.company_info,
        description="Use this tool to get general overview about a company, use this data to create a general \
                financial report. You should input the stock ticker to it"
    ),
    Tool(
         name="stock holders",
         func= agents_tools.get_stockholders,
         description="Use this tool to get information about stock holders, insider purchases, major holders, \
                mutual found holders, insider roster holders, institutional holders. You should input stock ticker to it"
    ),
    Tool(
        name="predict future prices",
        func=agents_tools.profit_forcasting,
        description="Use this to predict future stock pruces and trends. You should input stock ticker to it"
    ),
    Tool(
        name="DuckDuckGo Search",
        func=agents_tools.DuckDuck_search.run,
        description="Use when you need to get stock ticker from internet, you can also get recent stock related \
                    news to have insights about market ."
    ),
    # Tool(
    #     name="get recent news",
    #     func=agents_tools.get_recent_stock_news,
    #     description="Use this to fetch recent news about stocks and companies, or when you need general \
    #                 information about market news"
    # ),
    Tool(
        name="get financial statements",
        func=agents_tools.get_financial_statements,
        description="Use this to get financial statement of the company. With the help of this data companys historic \
                performance can be evaluated. You should input stock ticker to it"
    ),
    Tool(
        name="get quarterly financial statements",
        func=agents_tools.get_quarterly_financial_statements,
        description="Use this to get quarterly financial statement of the company. With the help of this data companys \
                historic performance can be evaluated. You should input stock ticker to it"
    ),
    Tool(
        name="google finance search",
        func=GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper()).run ,
        description="Use this to get stock market information. With the help of this data you can analys \
                market and financial status and get insights.",
    ),
    # Tool(
    #     name="financial facts",
    #     func=fin_tools.fin_facts,
    #     description="Use this to get latest financial status related to shares and investments",
    # )
] 

system_templat = """You are a financial advisor. 
    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. 
    As a Financial assistant you are able to generate human-like text based on the input you receive, allowing you to engage in natural-sounding conversations and provide 
    responses that are coherent and relevant to the topic at hand.
    with lots of expertise in stock market analysis and investment strategies that is working for a super important customer.
    Given a chat history of the latest user question answer user's questions as best as you can.
    if the user asked about a specific stock/company, then go directly to the tools and decide what is the best tool to use with minimal data loading. 
    if user query did not mentioned any specific stock/company then decide if it is nessesary to use any tool to search for the best answer or you can handle it directly whithout any tool.    
    Everytime first you should identify the company name and get the stock ticker symbole, 
    If you was asked to compare companies performance search for every company alone then gather the collected information to have the general comparison.
    If you dont know the answer say you do not know. 
    be polite and give greeting to user.

    You have access to the following tools: 
    company overview: Use this only if the user asked about general overview or background of a compny, compine this tool results with the results of DuckDuckGo Search tool to fetch general information about query company. You should input the stock ticker to it"           
    DuckDuckGo Search: Use this to fetch recent news about stocks and companies, or when you need general information about market news.
    google finance search: Use this tool to get stock market information if the result of DuckDuckGo Search tool is not suffiecient. With the help of this data you can analys market and financial status and get insights.
    get stock data: Use when you are asked about stock prices or to evaluate or analyze a stock, do not use if you do not have any company name. This will output historic share price data. You should input the stock ticker to it.
    get financial statements: Use this to get financial statement of the company. With the help of this data companys historic performance can be evaluated. You should input stock ticker to it
    get quarterly financial statements: Use this tool only if you was asked about quarterly financial statement. You should input stock ticker to it"
    stock holders: Use this tool if you was asked to get information about stock holders, insider purchases, major holders, mutual found holders, insider roster holders, institutional holders. You should input stock ticker to it.



    Use the following format:
    Question: the input question you must answer
    Conversation history: use the previos questions as conversation_history add it to the analysis so far
    Thought: you should always think about what to do, Also try to follow steps mentioned above
    Action: the action to take, should be one of [company overview, DuckDuckGo Search, google finance search, get stock data, get quarterly financial statements, get financial statements, stock holders]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    Begin!"""
    # Question: {input}
    # Ticker: {company}
    # Thought:{agent_scratchpad}
    # Output:{final_thoughts}

system_message = SystemMessagePromptTemplate.from_template(system_templat)

if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferWindowMemory(
        k = 2,
        memory_key="chat_history", 
        return_messages=True, 
        input_key="input", 
        output_key="output",
        )
    
agent_chain = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
    verbose=True, 
    memory=st.session_state['memory'], # ConversationBufferWindowMemory(k=1)
    max_iteration=5,
    max_execution_time=600,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    agent_kwargs={
        'system_message':system_templat,
        },
    max_token_limit = 10000,
    )
        
print('*****************************************************')
print('constracting memory: ', st.session_state['memory'].chat_memory.messages)
print('*****************************************************')

def handle_user_input(user_input):
    
    try:
        response = agent_chain.invoke({"input": user_input, 'chat_history': st.session_state['memory'].chat_memory}) #({"input": user_input}, config={"configurable": {"session_id": "abc12345"}})
        res = markdown.markdown(response['output'])
        res1 = res.replace('<p>', '')
        res1 = res1.replace('</p>', '')
        return res1
    except groq.RateLimitError as e:
        retry_after = 1
        st.write(f"Rate limit exceeded. Retrying in {retry_after} seconds...")
        time.sleep(retry_after)
    except Exception as e:
        st.write(f"An unexpected error occurred: {e}")
    return "Sorry, I couldn't process your request due to rate limits. Please refresh your window or try again later."

if 'responses' not in st.session_state:
        st.session_state['responses'] = ["Hello! I'm your financial assistant, specializing in stock market analysis and investment strategies. \
                                        I can remember your last three questions to provide better assistance. How can I help you today?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

### Statefully manage chat history ###

response_container = st.container()
textcontainer = st.container()

with textcontainer:
    with st.form("Question",clear_on_submit=True):
        # placeholder = st.empty()
        # query2 = placeholder.text_input("Add a question", key=1)
        query2 = st.text_input("Add a question")
        submit_button = st.form_submit_button(label = 'Submit')
        if submit_button:
            if query2 is not None and query2 != '':
                with st.spinner("thinking..."):
                    response = handle_user_input(query2)    
                    # memory.save_context({"input": query2}, {"output": response})
                    # memory.chat_memory.add_user_message(query2)
                    # memory.chat_memory.add_ai_message(response)
                    print('*****************************************************')
                    print(st.session_state['memory'].chat_memory.messages)
                    print('*****************************************************')
                    st.session_state.requests.append(query2)
                    st.session_state.responses.append(response)

        
with response_container:
    if st.session_state['responses']:
        # for i in range(100, 100 + len(st.session_state['responses2'])):
        #     message(st.session_state['responses2'][i-100],key=str(i))
        #     if i-100 < len(st.session_state['requests2']):
        #         message(st.session_state["requests2"][i-100], is_user=True,key=str(i)+ '_user')

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')


# getting stock symbol 
companies_names_file = pd.read_csv('edgar_companies_names.csv')
company_name = st.selectbox('Select a company', companies_names_file['title'])
symbol = companies_names_file.loc[companies_names_file['title'] == company_name, 'ticker'].values[0]
st.write(symbol)

# @st.cache_data
# def loading_data():
data, _, _, _, _, _ = agents_tools.get_recent_data(symbol)
if type(data.index) == pd.core.indexes.datetimes.DatetimeIndex:
    data.index = [str(x).split(" ")[0] for x in data.index]
    data.reset_index(inplace=True)
df_train = data[['index','Close']]
df_train = df_train.rename(columns={"index": "ds", "Close": "y"})

    # return [data, df_train]

# data, df_train = loading_data()
n_month = st.slider('Months of prediction:', 3, 24)
period = n_month * 30

st.subheader('Current prices')
# st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['index'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['index'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
# st.subheader('Forecast data')
# st.write(forecast.tail())
    
st.subheader(f'Forecast plot for {n_month} months')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
