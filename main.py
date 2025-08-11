from dotenv import load_dotenv
from huggingface_hub import login
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains import create_retrieval_chain

class brother:
    
    def __init__(self):
        
        ### Import the environment
        #from dotenv import load_dotenv
        
        load_dotenv()


        
        ### login to huginngface account
        #from huggingface_hub import login
        #import os
        
        login(os.getenv("HUGGINGFACE_HUB_TOKEN"))
        


        ### Embedding model download
        #from langchain_community.embeddings import SentenceTransformerEmbeddings
        
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


        ### Importing the pinecone vector DB and definning the rerriever
        #from langchain_pinecone import PineconeVectorStore
        
        import_index = PineconeVectorStore.from_existing_index(
        index_name= "sri-lanka-informations",
        embedding= embedding_model
        )

        retriever = import_index.as_retriever(search_type = "similarity", search_kwargs = {"k":3})



        ### Download the LLM from huggingface
        #from transformers import pipeline
        
        LLM_name = "google/flan-t5-base"
        chat_model = pipeline(task = "text2text-generation",
                              model = LLM_name,
                              max_new_tokens = 250,    # control answer length
                              temperature = 0.2,       # Lower = more factual answers
                              device_map = "auto",     # automaticaly select GPU if available
                              torch_type = "auto"      # Uses correct precision for your hardware
                              )


        
        ### make downloaded LLM from hugging face as langchin-fomfatable object
        #from langchain_huggingface import HuggingFacePipeline
       
        chatmodel = HuggingFacePipeline(pipeline=chat_model)



        ### Defining the system prompt
        #from langchain.prompts import PromptTemplate
        
        system_prompt = """
        You are an assistant that answers questions about Sri Lanka.

        Instructions:
        1. Use ONLY the information from the document 'Introduction to Sri Lanka.
        2. The document covers: geography, history, culture, economy, religion, transportation, health care, education, and society.
        3. Do NOT use outside knowledge or make up information.
        4. Your answer must be give from complete and clear sentences.
        5. If the answer is not found, reply "The document does not provide this information.".

        Context:
        {context}
        
        Question:
        {input}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "input"],
            template=system_prompt.strip()
            )
        
        
        
        ### Cobining the retrievers, qestion, and system prompt for LLM's input
        #from langchain.chains.combine_documents import create_stuff_documents_chain 
        
        question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)



        ### Connecting the pinecone vector DB and LLM through retriever
        #from langchain.chains import create_retrieval_chain
        
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)


        
        # Warm-up step — runs only once at startup
        try:
            print("Warming up retriever and LLM...")
            self.rag_chain.invoke({"input": "Hi"})
            print("Warm-up complete...")
        except Exception as error:
            print(f"Warm-up failed: {error}")
            
        ### The try and except block in Python is used for error handling
        
        ### What happens here?
        ### try block:
        ###   The program tries to execute the code inside this block.
        ###   It prints "Warming up retriever and LLM...".
        ###   Then it calls self.rag_chain.invoke({"input": "Hi"}) — probably this runs a "warm-up" call to a language model or retriever to prepare it.
        ###   If everything goes well, it prints "Warm-up complete ".
        ### except block:
        ###   If any error occurs while running the code in the try block, Python immediately stops there and jumps to this except block.
        ###   The variable e captures the error (exception) details.
        ###   It prints "Warm-up failed: <error message>", where <error message> is the actual problem message.    


    








