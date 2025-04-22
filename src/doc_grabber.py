from langchain_community.document_loaders import WebBaseLoader

def load_doc(url="https://en.wikipedia.org/wiki/Wikipedia:Unusual_articles"):
    loader = WebBaseLoader(url)
    return loader.load()


if __name__=='__main__':
    # debug stuff
    print(load_doc())
