# pip install -U duckduckgo-search

import requests
import os
import json
import re
from sec_api import QueryApi
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import yfinance as yf

from unstructured.partition.html import partition_html
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults, BaseTool, QuerySQLDataBaseTool
from functools import lru_cache
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go



embedder = HuggingFaceEmbeddings()

# @st.cache_resource(show_spinner=False)
@lru_cache(maxsize=128)
def get_stock(ticker):
    try:
      stock = yf.Ticker(ticker)
      if stock.history(period="1y").empty:
          ticker=ticker+".F"
          stock = yf.Ticker(ticker)
    except:
      print("Can't find this ticker!")
    return stock

@lru_cache(maxsize=1024)
def company_info(ticker):
    stock = get_stock(ticker)
    info = stock.info
    # info_df = pd.DataFrame(info)
    return info

# Fetch stock data from Yahoo Finance
# @st.cache_data(show_spinner=False)

@lru_cache(maxsize=2048)
def get_recent_data(ticker, period='5y'):

    stock = get_stock(ticker)
    latest_data =  stock.history(period=period)
    data = latest_data[['Close', 'Volume']]
    recent_news = stock.news
    calender = stock.calendar
    fast_info = stock.get_fast_info().items()
    recomendation_summary = stock.get_recommendations_summary()
    upgrades_downgrades = stock.get_upgrades_downgrades()
    # data.index=[str(x).split()[0] for x in list(data.index)]
    # data.index.rename("Date",inplace=True)
    return (latest_data,recent_news, calender, fast_info, recomendation_summary, upgrades_downgrades) #data


def get_stockholders(ticker):
    stock = get_stock(ticker)
    insider_purchases = stock.get_insider_purchases()
    major_holders = stock.get_major_holders()
    mutual_found_holders = stock.get_mutualfund_holders()
    insider_roster_holders = stock.insider_roster_holders
    institutional_holders = stock.institutional_holders
    return (insider_purchases, major_holders, mutual_found_holders, insider_roster_holders, institutional_holders)


@lru_cache(maxsize=2048)
def get_financial_statements(ticker):
    stock = get_stock(ticker)
    cashflow = stock.cashflow
    balance_sheet = stock.balance_sheet
    finance = stock.financials
    earnings_dates = stock.earnings_dates
    return (cashflow, balance_sheet, finance, earnings_dates)


def get_quarterly_financial_statements(ticker):
    stock = get_stock(ticker)
    quarterly_cashflow = stock.quarterly_cashflow
    quarterly_balance_sheet = stock.quarterly_balance_sheet
    quarterly_finance = stock.quarterly_financials
    return (quarterly_cashflow,quarterly_balance_sheet, quarterly_finance)


def profit_forcasting(ticker):
    data, _, _, _, _,_ = get_recent_data(ticker)
    if type(data.index) == pd.core.indexes.datetimes.DatetimeIndex:
        data.index = [str(x).split(" ")[0] for x in data.index]
        data.reset_index(inplace=True)
    df_train = data[['index','Close']]
    df_train = df_train.rename(columns={"index": "ds", "Close": "y"})

    period = 365

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    # fig1 = plot_plotly(m, forecast)
    # fig2 = m.plot_components(forecast)
    return (forecast)


# search tools:
def search_internet(query):
    """Useful to search the internet 
    about a a given topic and return relevant results"""
    top_result_to_return = 10
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': SERPER_KEY, #os.environ['SERPER_API_API_KEY'],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json() #['organic']
    print('==================================',results)
    string = []
    for result in results[:top_result_to_return]:
        try:
            string.append('\n'.join([
                f"Title: {result['title']}", f"Link: {result['link']}",
                f"Snippet: {result['snippet']}", "\n-----------------"
            ]))
        except KeyError:
            next

    return '\n'.join(string)

def search_news(query):
    """Useful to search news about a company, stock or any other
    topic and return relevant results"""""
    top_result_to_return = 10
    url = "https://google.serper.dev/news"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': SERPER_KEY, # os.environ['SERPER_API_API_KEY'],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()['news']
    print('==================================',results)
    string = []
    for result in results[:top_result_to_return]:
        try:
            string.append('\n'.join([
                f"Title: {result['title']}", f"Link: {result['link']}",
                f"Snippet: {result['snippet']}", "\n-----------------"
            ]))
        except KeyError:
            next

    return '\n'.join(string)

# SEC search tools:
def search_10q(data):
    """
    Useful to search information from the latest 10-Q form for a
    given stock.
    The input to this tool should be a pipe (|) separated text of
    length two, representing the stock ticker you are interested and what
    question you have from it.
		For example, `AAPL|what was last quarter's revenue`.
    """
    stock, ask = data.split("|")
    queryApi = QueryApi(api_key=os.environ['SEC_API_KEY'])
    query = {
        "query": {
            "query_string": {
            "query": f"ticker:{stock} AND formType:\"10-Q\""
            }
        },
        "from": "0",
        "size": "1",
        "sort": [{ "filedAt": { "order": "desc" }}]
    }

    fillings = queryApi.get_filings(query)['filings']
    if len(fillings) == 0:
        return "Sorry, I couldn't find any filling for this stock, check if the ticker is correct."
    link = fillings[0]['linkToFilingDetails']
    answer = embedding_search(link, ask)
    return answer

def search_10k(data):
    """
    Useful to search information from the latest 10-K form for a
    given stock.
    The input to this tool should be a pipe (|) separated text of
    length two, representing the stock ticker you are interested, what
    question you have from it.
    For example, `AAPL|what was last year's revenue`.
    """
    stock, ask = data.split("|")
    queryApi = QueryApi(api_key=os.environ['SEC_API_KEY'])
    
    query = {
        "query": {
            "query_string": {
            "query": f"ticker:{stock} AND formType:\"10-K\""
            }
        },
        "from": "0",
        "size": "1",
        "sort": [{ "filedAt": { "order": "desc" }}]
    }

    fillings = queryApi.get_filings(query)['filings']

    print('===============================')
    print('fillings' ,fillings)
    if len(fillings) == 0:
        return "Sorry, I couldn't find any filling for this stock, check if the ticker is correct."
    link = fillings[0]['linkToFilingDetails']
    answer = embedding_search(link, ask)
    return answer

def embedding_search(url, ask):
    text = download_form_html(url)
    elements = partition_html(text=text)
    content = "\n".join([str(el) for el in elements])
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 150,
        length_function = len,
        is_separator_regex = False,
    )
    docs = text_splitter.create_documents([content])
    retriever = FAISS.from_documents(
        docs, embedder
    ).as_retriever()
    answers = retriever.get_relevant_documents(ask, top_k=4)
    answers = "\n\n".join([a.page_content for a in answers])
    return answers

def download_form_html(url):
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7',
        'Cache-Control': 'max-age=0',
        'Dnt': '1',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"macOS"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    return response.text


DuckDuck_search=DuckDuckGoSearchRun(handle_tool_error=True, handle_validation_error=True)
DuckDuck_result = DuckDuckGoSearchResults(handle_tool_error=True, handle_validation_error=True)


# print('Duck search: \n\n',DuckDuck_search.run('amazon stock news'))
# print('Duck result: \n\n',DuckDuck_result.run('amazon stock news'))

def google_query(search_term):
    if "news" not in search_term:
        search_term=search_term+ "stock news " + "current market conditions and future growth prospects"
    url=f"https://www.google.com/search?q={search_term}"
    url=re.sub(r"\s","+",url)
    return url

    

# @st.cache_resource
def get_recent_stock_news(company_name):
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}

    g_query=google_query("market news and trends"+ company_name)
    res=requests.get(g_query,headers=headers).text
    soup=BeautifulSoup(res,"html.parser")
    news=[]
    for n in soup.find_all("div","kb0PBd cvP2Ce A9Y9g"):
        news.append(n.text)
    for n in soup.find_all("VwiC3b yXK7lf lVm3ye r025kc hJNv6b Hdw6tb"):
        news.append(n.text)

    if len(news)>10:
        news=news[:10]
    else:
        news=news
    news_string=""
    for i,n in enumerate(news):
        news_string+=f"{i}. {n}\n\n"
    top5_news="Recent News:\n\n"+news_string 

    return top5_news


def cik_matching_ticker(ticker):
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    ticker = ticker.upper().replace(".", "-")
    ticker_json = requests.get(
        "https://www.sec.gov/files/company_tickers.json", headers=headers
    ).json()

    for company in ticker_json.values():
        if company["ticker"] == ticker:
            cik = str(company["cik_str"]).zfill(10)
            return cik
    raise ValueError(f"Ticker {ticker} not found in SEC database")


def get_ticker_name(company_name):
    file = open('companies_names_from_edgar.json')
    company_list = json.load(file)
    ticker = None
    for element in company_list.values():

        if company_name.lower() in element['title']:
            print(element['title'])
            ticker = element['ticker']
            break
    return ticker


def get_facts(ticker):
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    cik = cik_matching_ticker(ticker)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    company_facts = requests.get(url, headers=headers).json()
    return company_facts


import pandas as pd

def fin_facts(ticker):
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    facts = get_facts(ticker, headers)
    us_gaap_data = facts["facts"]["us-gaap"]
    df_data = []
    for fact, details in us_gaap_data.items():
        for unit in details["units"]:
            for item in details["units"][unit]:
                row = item.copy()
                row["fact"] = fact
                df_data.append(row)

    df = pd.DataFrame(df_data)
    df["end"] = pd.to_datetime(df["end"])
    df["start"] = pd.to_datetime(df["start"])
    df = df.drop_duplicates(subset=["fact", "end", "val"])
    df.set_index("end", inplace=True)
    labels_dict = {fact: details["label"] for fact, details in us_gaap_data.items()}
    return df, labels_dict



# print('*******************************************')
# print('ticler fact: \n\n', get_facts('313838'))

# print('*******************************************')
# search_internet('amazon')

# print('*******************************************')
# search_news('Amazon')

# print('*******************************************')
# search_10k('AAPL|what was last years revenue')
