# finance-ai-assistant
Ai Agent that writes reports and anasyses data showing visualisation on the Frontend

AI workflow (agent in the future) 
for finance reports for companies that could read the data
and alalyse it, summarizing, creating visualization, and writing a report.
if possible, it can be viewed in Fronted like streamlit or a Flask app

1. Input: Collect or Load Finance Reports 
2. Preprocessing: Clean and Extract Structured Data 
3. Embedding: Turn Data into Vectors (Embedding Ollama Model)
4. Retrieval (Optional): Search Specific Info in Large Reports (Milvus)
5. LLM Interaction: Summarize / Generate Analysis (qwen chat model)
6. Saving history in the json file
7. Showing it within basic Frontend
8. Visualizing the data withing graphs
