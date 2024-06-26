import mysql.connector
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers , HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM , AutoTokenizer , pipeline , BitsAndBytesConfig
import torch
import time

def login_validation_check(number , password, type):
    conn = mysql.connector.connect(
        host='localhost',  # e.g., 'localhost'
        user='root',  # e.g., 'root'
        password='Siddha@2234',  # e.g., 'password'
        database='agribot'  # e.g., 'mydatabase'
    )
    cursor = conn.cursor()
    if type=="farmer":
        query = "SELECT password FROM farmers WHERE mobile_number = %s"
        cursor.execute(query, (number,))
        result = cursor.fetchone()
        if result[0]==password:
            return True
        else:
            return False
    else:
        query = "SELECT password FROM buyer WHERE mobile_number = %s"
        cursor.execute(query, (number,))
        result = cursor.fetchone()
        if result[0]==password:
            return True
        else:
            return False
        
def selling_injection_in_sql(name, email, contact, address, product_name, product_type, quantity, price, unique_id, password):
    conn = mysql.connector.connect(
        host='localhost',  # e.g., 'localhost'
        user='root',  # e.g., 'root'
        password='Siddha@2234',  # e.g., 'password'
        database='agribot'  # e.g., 'mydatabase'
    )
    cursor = conn.cursor()
    insert_query = """
            INSERT INTO selling (name, EmailID, contact_number, locality_address, product_name, product_quantity, unique_id, price, password, prodcut_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, (name, email, contact, address, product_name, quantity,unique_id, price, password,product_type))
    conn.commit()
    return True

def generate_response(user_input,type_of_llm,category):
    if type_of_llm=='1':
        #docs rag
        
        PROMPT_TEMPLATE = '''
        With the information provided try to answer the question. 
        If you cant answer the question based on the information either say you cant find an answer or unable to find an answer.
        So try to understand in depth about the context and answer only based on the information provided. Dont generate irrelevant answers

        Context: {context}
        Question: {question}
        Do provide only helpful answers

        Helpful answer:
        '''
         
        INP_VARS = ['context', 'question']
        custom_prompt_template = PromptTemplate(
            template = PROMPT_TEMPLATE,
            input_variables = INP_VARS
        )
        llm = CTransformers(
            model = r"D:\Anurag-Agri-Bot\llms\llama-2-7b-chat.ggmlv3.q4_1.bin",
            model_type="llama",
            max_new_tokens = 512,
            temperature = 0.1,
            callbacks=[StreamingStdOutCallbackHandler()],
            gpu_layers=0
        )
        hfembeddings = HuggingFaceEmbeddings(
            model_name = "thenlper/gte-large",
            model_kwargs = {'device':'cuda'}
        )
        vector_db = FAISS.load_local(r"D:\Anurag-Agri-Bot\datas\faiss\agri_data/", hfembeddings,allow_dangerous_deserialization=True)
        retrieval_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={'k': 1}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt_template}
        )
        prompt = {'query': user_input}
        model_out = retrieval_qa_chain(prompt)
        answer = model_out['result']
        return answer
    

    elif type_of_llm=='2':
        #web scraper rag

        PROMPT_TEMPLATE = '''
        With the information provided try to answer the question. 
        If you cant answer the question based on the information either say you cant find an answer or unable to find an answer.
        So try to understand in depth about the context and answer only based on the information provided. Dont generate irrelevant answers

        Context: {context}
        Question: {question}
        Do provide only helpful answers

        Helpful answer:
        '''
        INP_VARS = ['context', 'question']
        custom_prompt_template = PromptTemplate(
            template = PROMPT_TEMPLATE,
            input_variables = INP_VARS
        )
        llm = CTransformers(
            model = r"D:\Anurag-Agri-Bot\llms\llama-2-7b-chat.ggmlv3.q4_1.bin",
            model_type="llama",
            max_new_tokens = 512,
            temperature = 0.1,
            callbacks=[StreamingStdOutCallbackHandler()],
            gpu_layers=0
        )
        hfembeddings = HuggingFaceEmbeddings(
            model_name = "thenlper/gte-large",
            model_kwargs = {'device':'cuda'}
        )
        vector_db = FAISS.load_local(r"D:\Anurag-Agri-Bot\datas\faiss\custum_website/", hfembeddings,allow_dangerous_deserialization=True)
        retrieval_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={'k': 1}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt_template}
        )
        prompt = {'query': user_input}
        model_out = retrieval_qa_chain(prompt)
        answer = model_out['result']
        return answer
    

    elif type_of_llm=='3':
        # database administrator
        if category == "animals_details":
            context = "create table animals_details(cattlename varchar(30), quantity integer(5));"
        elif category == "cattle":
            context = "create table cattle(name varchar(50), sellername varchar(50), price integer(10), quantity integer(10), locality varchar(100));"
        elif category == "details":
            context = "create table details(acre integer(30), current_crop varchar(30), soil_type varchar(30), fertilizer_name varchar(100), fertilizer_company varchar(100), equipments_name varchar(100), equipments_quantity integer(5), fertilizer_type varchar(100), labour_used integer(5), seed varchar(30));"
        elif category == "fertilizer":
            context = "create table fertilizer(name varchar(50), sellername varchar(50), usedfor varchar(60), quantity integer(10), price integer(10));"
        elif category == "financial":
            context = "create table financial(loanid varchar(20), userid varchar(20), loantype varchar(30), loanamount integer(10), interestrate varchar(10), loanterm varchar(30), applicationdate varchar(20), approvaldate varchar(20), loanstatus varchar(30), repaymentschedule varchar(20), expirationdate varchar(20), policystatus varchar(30), insurancetype varchar(60), coverageamount integer(10), policyterm varchar(30), policyid varchar(20), issuancedate varchar(20), coveragedetails varchar(60));"
        elif category == "insurance":
            context = "create table insurance(insurancetype varchar(60), insurancepolicyname varchar(100), duration varchar(30), companyname varchar(50), amount varchar(30));"
        elif category == "loan":
            context = "create table loan(loanname varchar(50), loantype varchar(50), interestrate varchar(30), bankname varchar(50), duration varchar(30));"
        elif category == "machinery":
            context = "create table machinery(name varchar(50), sellername varchar(50), price integer(10), quantity integer(10));"
        elif category == "manufacturer":
            context = "create table manufacturer(name varchar(100), manufacturer_id varchar(10), mobile_number integer(10), company_name varchar(100), email varchar(30), password varchar(20), type varchar(30));"
        elif category == "personal_details":
            context = "create table personal_details(name varchar(50), email varchar(20), address varchar(60), age integer(5), state varchar(20), pincode integer(10), mobilenumber integer(20));"
        elif category == "purchase_history":
            context = "create table purchase_history(product varchar(30), price integer(10), quantity integer(5), dateofpurchase varchar(20), insurancepolicyname varchar(60), insuranceduration varchar(20), insuranceissuancedate varchar(20), insuranceamount integer(10), loanname varchar(60), loanamount integer(10), loanduration varchar(20), loanissuancedate varchar(20));"
        elif category == "rental":
            context = "create table rental(name varchar(50), price integer(10), sellername varchar(50));"
        elif category == "seed":
            context = "create table seed(name varchar(50), type varchar(20), sellername varchar(50), quantity integer(10), price integer(10));"
        elif category == "selling":
            context = "create table selling(name varchar(50), EmailID varchar(20), contact_number integer(20), locality_address varchar(100), product_name varchar(50), product_quantity integer(20), unique_id varchar(20), price integer(10), password varchar(20), prodcut_type varchar(30));"
        else:
            pass


        model_id = "siddharth-magesh/Tiny_Lllama-AgriDB"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map = "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokenizer.pad_token = tokenizer.eos_token
        pipe = pipeline(
            "text-generation",
            model = model,
            tokenizer = tokenizer,
        )
        llm = HuggingFacePipeline(
            pipeline = pipe,
            pipeline_kwargs={
                "temperature": 0.3,
                "max_new_tokens": 16,
                "min_length": 8,  # Ensure a minimum length
                "do_sample": True,
                "num_beams": 5,
                "repetition_penalty": 2.0,  # Penalize repetition
                "no_repeat_ngram_size": 3,   # Use beam search for better long outputs
            }
        )
        PROMPT_TEMPLATE = """\
        <|im_start|>user
        Given the context, generate an SQL query for the following question
        context:{context}
        question:{question}
        <|im_end|>
        <|im_start|>assistant
        """
        prompt = PromptTemplate(template=PROMPT_TEMPLATE,input_variables=["context","question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        answer = llm_chain.run(context=context,question=user_input)
        return answer
        

    elif type_of_llm=='4':
        #general chatbot

        llm = HuggingFacePipeline.from_model_id(
            model_id="siddharth-magesh/Tiny-Llama-Agri-Bot",
            task="text-generation",
            pipeline_kwargs={
                "temperature": 0.3,
                "max_new_tokens": 256,
                "min_length": 16,  # Ensure a minimum length
                "do_sample": True,
                "num_beams": 5,
                "repetition_penalty": 2.0,  # Penalize repetition
                "no_repeat_ngram_size": 3,   # Use beam search for better long outputs
            },
            device = 0,
        )
        template = """ Question: {question} Answer the following question ###Answer : """
        prompt = PromptTemplate(template=template,input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        x=llm_chain.run(user_input)
        return x

def signup_sql_injection(name, mobile_number, password, address, gender, age, dateofbirth, email, blood_group, unique_id, state, country):
    db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Siddha@2234',
    database='agribot'
    )       
    cursor = db.cursor()
    
    cursor = db.cursor()

    # Prepare SQL query to insert a record into the database
    sql = "INSERT INTO farmer1 (name, mobile_number, password, address, gender, age, dateofbirth, email, blood_group, unique_id, state, country) " \
          "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

    # Execute the SQL query
    cursor.execute(sql, (name, mobile_number, password, address, gender, age, dateofbirth, email, blood_group, unique_id, state, country))

    # Commit changes to the database
    db.commit()

    # Close the cursor
    cursor.close()


def compute_plan_agri(landMeasurements,budget,machinery,labours,soilType,irrigationMethod,storageFacilities):
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "temperature": 0.3,
            "max_new_tokens": 256,
            "min_length": 16,  # Ensure a minimum length
            "do_sample": True,
            "num_beams": 5,
            "repetition_penalty": 2.0,  # Penalize repetition
            "no_repeat_ngram_size": 3,   # Use beam search for better long outputs
        },
        device = 0,
    )
    template = """You are a agricultural chatbot which summarizes a work plan by going through the following details
    landMeasurements : {landMeasurements}
    budget : {budget}
    machinery : {machinery}
    labours : {labours}
    soilType : {soilType}
    irrigationMethod : {irrigationMethod}
    storageFacilities : {storageFacilities}
    generate a detailed work plan by using these informations
      """
    prompt = PromptTemplate(template=template,input_variables=["landMeasurements","budget","machinery","labours","soilType","irrigationMethod","storageFacilities"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    x=llm_chain.run(landMeasurements=landMeasurements,budget=budget,machinery=machinery,labours=labours,soilType=soilType,irrigationMethod=irrigationMethod,storageFacilities=storageFacilities)
    return x