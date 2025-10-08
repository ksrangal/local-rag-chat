import sys 
import os
import time
from ingester import Ingester

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama 
from langchain_chroma import Chroma

"""
Main entry point for the Streamlit Local RAG Chat application.
This function configures the Streamlit page and UI, accepts user-provided inputs from the
sidebar (a vector DB / data directory and a prompt template), validates those inputs,
runs a local ingestion -> embedding -> vector-store pipeline, initializes an embedding
and chat model, constructs a retrieval + generation chain, and answers user queries in
the Streamlit chat interface.
"""
def main(): 
  st.set_page_config(page_title="Local AI Assistant", page_icon="🤖")
  st.title("AI Assistant — Your Local RAG Chat")

  with st.sidebar: 
    st.header("Configure")
    data_dir = st.text_input("Enter RAG data directory")

    if not data_dir:
      st.warning("Please enter a valid vector DB path to continue.")
      st.stop()

    if not os.path.exists(data_dir):
      st.error(f"'{data_dir}' does not exist")
      st.stop()

    st.divider()
    st.markdown("**Prompt Template:**")
    template = st.text_area("Provide prompt template:")

    if not template: 
      st.warning("Please enter a prompt template: \n\n"
                 "Example: \n"
                 "I am a magician\n\n"
                 "Here are some relevant reviews: {reviews} \n\n"
                 "Here is the question to answer: {question}\n\n")
      st.stop()


  start_time = time.time()

  # chuck the data 
  ingester = Ingester(data_dir)
  ingester.chunck()
 
  # Embed and store in the vector chroma db
  ingester.store()

  end_time = time.time()
  print(f"Data ingestion completed in {end_time - start_time:.2f} seconds.")

  # Now user can start chatting. 
  embedding = OllamaEmbeddings(model="mxbai-embed-large")
  vector_db = Chroma(embedding_function=embedding, persist_directory="./chroma.db")
  ollama_model = ChatOllama(model="gemma3")

  retriever = vector_db.as_retriever(search_type="similarity",search_kwargs={"k": 5})

  prompt = ChatPromptTemplate.from_template(template)

  output_parser = StrOutputParser()

  chain = prompt | ollama_model | output_parser

  query = st.chat_input("Ask:")
  if query:
    documents  = retriever.invoke(query)
    reviews = "\n\n".join([d.page_content for d in documents])
    output = chain.invoke({"reviews": reviews, "question": query})
    st.chat_message("assistant").write(output)

if __name__=="__main__":
  main() 
