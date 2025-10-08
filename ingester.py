import sys 
import os
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveJsonSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

"""
  Ingester
  High-level utility for loading, chunking, and storing document embeddings into a Chroma vector store.
  Attributes
  ----------
  data_dir : str
    Path to the directory containing source files to ingest (expects .pdf and/or .json files).
  vector_db_path : str
    Filesystem path used to persist the Chroma vector store (default "./chroma.db").
  chunks : list
    In-memory list of chunked documents ready for embedding; populated after calling chunck().
  embeddings : OllamaEmbeddings
    Embedding model instance used to convert text chunks into vectors.
  vector_db : Chroma | None
    Chroma vector store instance once created or loaded via store().

  Example
  -------
  ingester = Ingester("/path/to/data")
  ingester.chunck()   # populate ingester.chunks
  ingester.store()    # create or load persistent vector store
"""
class Ingester:
  def __init__(self, data_dir):
    self.data_dir = data_dir 
    self.vector_db_path = "./chroma.db"
    self.chunks = None
    self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    self.vector_db = None

  def chunck_data(self, file_path, file_type, chunk_size=1000, chunk_overlap=200):
    print(f"[INFO] chuncking: {file_path}")
    if file_type == "pdf":
      # Load the PDF document
      loader = PyPDFLoader(file_path)
      documents = loader.load()

      # Split the document into chunks
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
      chunks = text_splitter.split_documents(documents)

    elif file_type == "json":
      json_splitter = RecursiveJsonSplitter(max_chunk_size=chunk_size)
      with open(file_path, 'r') as j_file:
        json_data = json.load(j_file)

      chunks = json_splitter.create_documents(texts=[json_data])
      # chunks = json_splitter.split_documents(documents)

    else:
      raise NotImplementedError(f"{file_type} not supported")

    return chunks

  """
  Create or load the Chroma vector store and persist embeddings.
    Behavior
    --------
    - If a persisted vector DB exists at vector_db_path, the existing store is loaded.
    - Otherwise, embeddings are generated from self.chunks and a new Chroma store is created and persisted.
    Notes
    -----
    - Expects self.chunks to be populated before calling store().
    - After creation/loading, the method prints the number of stored embeddings.
  """
  def store(self):

    # Create or load the vector store index
    if os.path.exists(self.vector_db_path):
        print("[INFO] Vectior DB already exist. !!!")
        self.vector_db = Chroma(embedding_function=self.embeddings, persist_directory=self.vector_db_path)
    else:
        self.vector_db = Chroma.from_documents(documents=self.chunks,
                                             embedding=self.embeddings,
                                             persist_directory=self.vector_db_path)

    print(f"Number of embeddings: {self.vector_db._collection.count()}\n")
 

  """
    chunck()
    Discover supported files in data_dir, chunk each file, and aggregate chunks into self.chunks.
    Behavior
    --------
    - Scans data_dir for files ending in .pdf and .json.
    - Uses chunck_data() to split each file into chunks.
    - Aggregates all chunks from all files into self.chunks for downstream embedding and storage.
    Raises
    ------
    FileNotFoundError
      If the provided data_dir does not exist, or if it contains neither .pdf nor .json files.
  """  
  def chunck(self):
    if not os.path.exists(self.data_dir):
      raise FileNotFoundError(f"\n[ERROR] '{self.data_dir}' does not exist")

    # get all the pdf json files. 
    pdf_files = []
    json_files = []
    for file in os.listdir(self.data_dir):
      file_fp = os.path.join(self.data_dir, file)
      if file.endswith(".pdf"):
        pdf_files.append(file_fp) 
      elif file.endswith(".json"):
        json_files.append(file_fp)
      else: 
        print(f"[WARN] ignoring: {file}")

    if not pdf_files and not json_files:
      raise FileNotFoundError(f"\n[ERROR] '{self.data_dir}' does not contain pdf or json")

    for pdf_file in pdf_files:
      pdf_chunks = self.chunck_data(pdf_file, "pdf")
      pdf_chunks += pdf_chunks

    for json_file in json_files:
      json_chunks = self.chunck_data(json_file, "json")
      json_chunks += json_chunks

    self.chunks = pdf_chunks + json_chunks
