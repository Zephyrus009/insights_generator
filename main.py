from flask import Flask,render_template,redirect,request
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
from dotenv import load_dotenv
load_dotenv()

def create_app():
    app = Flask(__name__)

    @app.route("/", methods=['GET', 'POST'])
    def home():
        return render_template('front.html')

    @app.route("/research", methods=['GET', 'POST'])
    def research():
        llm = OpenAI(temperature=0.9,max_tokens=500)
        url1 = request.form.get("url1")
        url2 = request.form.get("url2")
        url3 = request.form.get("url3")

        load_documents = UnstructuredURLLoader(urls=[url1,url2,url3])
        data = load_documents.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['/n/n','/n','.'],
            chunk_size = 1000
        )

        splitted_docs = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(splitted_docs,embeddings)
        vectorstore_openai

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore_openai.as_retriever())
        query = request.form.get('question')
        result = chain({'question':query},return_only_outputs=True)

        answer = result['answer']

        return render_template('front.html',
                              answer = answer)
    
    return app