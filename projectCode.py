from langchain_community.document_loaders import DirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
DataPath= "data/books"

def  LoadDocument():
    loaded = DirectoryLoader(DataPath, glob="**/*.md")
    document = loaded.load()
    return document

# we are splitting data so data our analysis becomes more focused and result are related to passed query
def DataSliptter(document):
    text_splitter =  RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks=text_splitter.split_documents(document)
    print(f"number of documents {len(document)} and number of chunk size {len(chunks)}")
    
CalledData=LoadDocument()

DataSliptter(CalledData)    