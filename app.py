import streamlit as st
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
with st.sidebar:
    st.title('RAG PDF QUERY APPLICATON')
    st.markdown('''
     ## About:
     This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/) PLATFORM TO RUN APP
    - [LangChain](https://python.langchain.com/) FRAMEWORK
    - [COHERE API](https://dashboard.cohere.com/) COMPUTE EMBEDDINGS
    - [GEMINI](https://gemini.google.com/) LLM MODEL''')
load_dotenv()
def main():
    st.header("Q&A BOT FOR PERSONAL PDF FILES")
    #Upload PDF:
    pdf=st.file_uploader("Upload your PDF here/-",type='pdf')
    if pdf:
        st.write(pdf.name)
    if pdf is not None:
        #PDF reader:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+= page.extract_text()
        #Divide Content into tokens:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)
        # #embeddings
        #store embedded vector in local device to save credits
        # store_name=pdf.name[:-4]
        # if os.path.exists(f"{store_name}.pkl"):
        #     with open(f"{store_name}.pkl","wb") as f:
        #         vectorstore = pickle.load(f)
        #     st.write('Embeddings loaded from disk')
        # else:
            # embeddings=OpenAIEmbeddings()
        embed_model = CohereEmbeddings()
        vectorstore=FAISS.from_texts(chunks,embedding=embed_model)
            # with open(f"{store_name}.pkl","wb") as f:
            #      pickle.dump(vectorstore,f)
        st.write('Embeddings Computed')
        # #Accept User Query
        query_=st.text_input("Ask questions related to your PDF")
        st.write(query_)
        if query_:
            docs=vectorstore.similarity_search(query=query_,k=3)
            #Answering through openai llm 
            # llms=OpenAI(model_name='gpt-3.5-turbo')
            llms = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
            chain = load_qa_chain(llm=llms,chain_type="stuff")
            with get_openai_callback() as CB:
                response = chain.run(input_documents=docs,question=query_)
                print(CB)
                st.write(CB)
            st.write(response)


if __name__=='__main__':
    main()