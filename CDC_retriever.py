from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
url = "https://www.cdc.gov"


def CDC_retriever():

    loader = RecursiveUrlLoader(
        url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs = loader.load()

    text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter2.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    CDC_retriever = vectorstore.as_retriever()

return CDC_retriever
