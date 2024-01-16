from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader

def docx2txt(path):#using this for model
    text = ""
    loader = Docx2txtLoader(path)
    pages = loader.load()
    for page in pages:
        text += page.page_content
    return text

def unstructured(path):
    text = ""
    loader = UnstructuredWordDocumentLoader(path)
    pages = loader.load()
    for page in pages:
        text += page.page_content
    return text
