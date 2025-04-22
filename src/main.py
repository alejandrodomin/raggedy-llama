from langchain_nomic import NomicEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

def grab_docs(vector_store):
    # some soupy curl grabber to get a random wikipedia page to start
    test_document = Document(page_content="test", metadata={"source": "me"})

    documents=[]
    documents.append(test_document)

    gen_uuids = lambda documents: [str(uuid4()) for _ in range(len(documents))]

    return vector_store.add_documents(documents=documents, ids=gen_uuids(documents))

def rag_docs():
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

    return FAISS(
        embedding_function=embeddings,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


if __name__=='__main__':
    # adding a skeleton of what I want to do
    grab_docs(rag_docs())

    start_model()

    ask_questions()
