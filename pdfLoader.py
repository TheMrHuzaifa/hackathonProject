from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyMuPDFLoader

def pypdf(path):
    text = ""
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    for page in pages:
        text += page.page_content
    return text

def pymupdf(path): #using this for model
    text = ""
    loader = PyMuPDFLoader(path)
    pages = loader.load()
    for page in pages:
        text += page.page_content
    return text
