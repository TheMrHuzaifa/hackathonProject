from zipfile import ZipFile
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfLoader import pymupdf
from docxLoader import docx2txt

def zip_loader(documnet):
    with ZipFile(documnet,'r') as ref:
        file_list = ref.namelist()

        for filename in file_list:
            if filename.endswith(".pdf"):
                with ref.open(filename, 'r') as file:
                   file_path = ref.extract(filename)
                   pdf_text = pymupdf(file_path)
                   pdf_chunk = get_text_chunk_char(pdf_text, filename)
                #    pdf_chunk = get_text_chunk_recursive(pdf_text, filename)

            elif filename.endswith(".docx"):
                with ref.open(filename, 'r') as file:
                    file_path = ref.extract(filename)
                    docx_text = docx2txt(file_path)
                    docx_chunk = get_text_chunk_char(docx_text, filename)
                    # docx_chunk = get_text_chunk_recursive(docx_text, filename)

            else:
                continue
    
    return pdf_chunk + docx_chunk

def get_text_chunk_char(text, file_name):
    text_spliter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 100,
        chunk_overlap = 20,
        length_function = len
    )

    chunks = text_spliter.split_text(text)

    doc_list = []
    for chunk in chunks:
        metadata = {"source":file_name}
        doc_string = Document(page_content=chunk, metadata=metadata)
        doc_list.append(doc_string)
    return doc_list

def get_text_chunk_recursive(text, file_name):
    text_spliter = RecursiveCharacterTextSplitter(
        # by default the separator is set to paragraph, sentence, word, character-level
        chunk_size = 100,
        chunk_overlap = 20,
        length_function = len
    )

    chunks = text_spliter.split_text(text)

    doc_list = []
    for chunk in chunks:
        metadata = {"source":file_name}
        doc_string = Document(page_content=chunk, metadata=metadata)
        doc_list.append(doc_string)
    return doc_list