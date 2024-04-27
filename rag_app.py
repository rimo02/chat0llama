import os
from langchain_community.embeddings import OllamaEmbeddings
import chainlit as cl
import PyPDF2
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain_pinecone import Pinecone
from langchain_community.chat_models import ChatOllama

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)

HUGGINGFACEHUB_API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_API_ENV = os.environ['PINECONE_API_ENV']


@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file",
            accept=['application/pdf'],
            max_size_mb=20,
            timeout=180,
        ).send()
    file = files[0]
    msg = cl.Message(content=f'Processing `{file.name}`...')
    await msg.send()

    # Read the PDF
    pdf = PyPDF2.PdfReader(file.path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

    texts = text_splitter.split_text(text)
    embeddings = OllamaEmbeddings(model="mistral")

    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, api_env=PINECONE_API_ENV)
    index_name = 'chatpdf'
    docsearch = await cl.make_async(Pinecone.from_texts)(
        texts, embeddings, index_name=index_name)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    # print(docsearch.as_retriever())
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model="mistral"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    # cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content)
    answer = res['answer']
    source_documents = res['source_documents']

    text_elements = []
    if source_documents:
        for idx, doc in enumerate(source_documents):
            source_name = f'source_{idx}'
            text_elements.append(cl.Text(
                content=doc.page_content,
                name=source_name
            ))
        source_names = [el.name for el in text_elements]

        if source_names:
            answer += f"\n Sources: {', '.join(source_names)}"
        else:
            answer += '\n No sources found'

    await cl.Message(content=answer, elements=text_elements).send()
