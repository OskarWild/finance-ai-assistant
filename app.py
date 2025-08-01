import os
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv

# Langahcain and Ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# MCP and AI Agent
from langchain.tools.render import render_text_description
from langgraph.prebuilt.chat_agent_executor import StructuredResponse
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import MCPTool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Environmental variables passing
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL")
LLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN")
MILVUS_URI = os.environ.get("MILVUS_URI")
COLLECTION_NAME = "finance_reports"

# Setting up Milvus
from pymilvus import MilvusClient

if not MILVUS_TOKEN:
    raise ValueError("Please set the MILVUS_TOKEN environment variable.")

# Connectingg
milvus_client = MilvusClient(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN,
)

# Create collection only if not exist
if not milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=768,
        metric_type="COSINE",
        consistency_level="Strong",
        auto_id=True,
        description="Embeddings of financial reports"
    ) # else: print(f"Collection '{COLLECTION_NAME}' already exists. Skipping the creation")

# Plotting the data as Bar Chart and Correlational Map
def plot_summary(df):
    st.subheader("Data Overview & Visualizations")
    st.write("Raw Table Snapshot:")
    st.dataframe(df.head(10))

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available for plotting.")
        return

    st.write("Max Values per Numeric Column")
    max_vals = df[numeric_cols].max()
    fig, ax = plt.subplots()
    max_vals.plot(kind='bar', ax=ax)
    ax.set_title("Maximum Values")
    st.pyplot(fig)

    if len(numeric_cols) >= 2:
        st.write("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.write("Custom Column Plot")
    col_to_plot = st.selectbox("Choose a column to plot (bar chart):", numeric_cols)
    fig, ax = plt.subplots()
    df[col_to_plot].dropna().plot(kind='bar', ax=ax, title=f"{col_to_plot} Distribution")
    st.pyplot(fig)

# loading a list of dataframs from the report folder (CSV or Exel)
def load_reports_from_folder(folder_path="./data"):
    reports = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            reports.append((df, file))
        elif file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(folder_path, file))
            reports.append((df, file))
    return reports

def load_report(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Extracting text from the dataframe
def dataframe_to_text(df, filename=None):
    header = f"Financial Report: {filename}\n\n"
    body = df.to_string(index=False)
    return header + body

# prompts for chat model, pasring output, returning the result
def summarize_report(text: str, query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial assistant. Carefully read the financial report and help the user with their request."),
        ("human", """
        This is a financial report:

        {report}

        The user asks: {query}

        Please answer as clearly and informatively as possible.
        
        """)
    ])

    chat_model = ChatOllama(model=OLLAMA_CHAT_MODEL, base_url="http://localhost:11434")
    chain = prompt | chat_model | StrOutputParser()
    return chain.invoke({"report": text, "query": query})

# Embedding and Storing
def embed_and_store(text: str, metadata: dict):
    embedder = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url="http://localhost:11434")
    vector = embedder.embed_query(text)

    milvus_client.insert(
        collection_name=COLLECTION_NAME,
        data=[{"vector": vector, "text": text, "filename": metadata.get("filename", "unknown")}]
        ) 

# REPORTS HISTORY IN JSON
HISTORY_PATH = "reports_history.jsonl"
if os.path.dirname(HISTORY_PATH):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

def log_to_history(filename, report):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "filename": filename,
        "report": report
    }
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    folder_path = "./data"
    filename = "Detailed_Financial_Report.csv"
    file_path = os.path.join(folder_path, filename)
    reports = load_reports_from_folder(folder_path)

    df = load_report(file_path) 

    if df is None:
        st.error(f"Failed to load {filename}")
        return

    assistant_name = "Financing Assistant with Analysis & Visuals"
    st.title(assistant_name)

    plot_summary(df) 

    user_query = st.text_input("Write you query for the assistant")
    user_query_1 = "Summarize the key financial metrics and trends in 5 bullet points."
    user_query_2 = "Name the headers and for each one, show their maximum values"

    if user_query:
        print("\nGot your query!")

        print(f"\nProcessing: {filename}")
        report_text = dataframe_to_text(df, filename)

        summary = summarize_report(report_text, user_query)

        print("Summary:\n", summary)
        st.write(summary)

        embed_and_store(text=report_text, metadata={"filename": filename})
        print("Stored in Milvus.")

        # Save Filname <-> Report in json
        log_to_history(filename, summary)

        st.success("Report processed and stored.")
        st.session_state.processed = True

if __name__ == "__main__":
    main()



# streamlit run app.py
# What are the top 3 years for metrics












    # if user_query:
    #     for df, filename in reports:
    #         print(f"\nProcessing: {filename}")
    #         report_text = dataframe_to_text(df, filename)

    #         summary = summarize_report(report_text, user_query)

    #         print("Summary:\n", summary)
    #         st.write(summary)

    #         embed_and_store(text=report_text, metadata={"filename": filename})
    #         print("Stored in Milvus.")

    #         # Save Filname <-> Report in json
    #         log_to_history(filename, summary)
