from dotenv import load_dotenv
import os
import streamlit as st
# from PyPDF2 import PdfReader
# import pdfminer
from pdfminer.high_level import extract_text
import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback



def main():
    load_dotenv(os.getenv("OPENAI_API_KEY"))
    st.set_page_config(page_title="Chat-PDF")
    st.header("Chat-PDF")

    # upload pdf
    pdf=st.file_uploader("Upload PDF", type=["pdf"])

    # extract text from pdf
    if pdf is not None:
        text = extract_text(pdf) 
        # st.write(text)

        # split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        # st.write(chunks)a

        with get_openai_callback() as cb:
            # create embeddings
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            print(cb)
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        user_question = st.text_input("Ask a question")

        langchain.verbose = False
        if user_question:
            docs = knowledge_base.similarity_search(user_question, k = 8)
            st.write(docs)

            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)
        


if __name__ == '__main__':
    main()