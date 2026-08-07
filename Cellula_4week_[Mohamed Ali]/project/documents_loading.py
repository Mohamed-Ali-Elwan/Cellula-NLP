import pandas as pd
from langchain_classic.document_loaders import CSVLoader,JSONLoader,TextLoader,UnstructuredWordDocumentLoader,UnstructuredPowerPointLoader,UnstructuredMarkdownLoader,PyPDFLoader
from langchain_core.documents import Document



base_data = pd.read_parquet("hf://datasets/openai/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")


documents = []
class DocumentLoader:
    @staticmethod
    def load_base_data():
        for row,sample in base_data.iterrows():
            text= f'''
             Task ID : {sample['task_id']} ,
             Prompt : {sample['prompt']} ,
             Solution : {sample['canonical_solution']} ,
             Test Cases : {sample['test']}
            '''
            documents.append(Document(page_content=text, metadata={"task_id": sample['task_id'] , "entry_point": sample['entry_point']}))
        return documents
