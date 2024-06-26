from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers , HuggingFaceHub
from langchain.chains import RetrievalQA
from speech import recognize_speech , text_to_speech

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
    max_new_tokens = 1024,
    temperature = 0.5
)
hfembeddings = HuggingFaceEmbeddings(
    model_name = "thenlper/gte-large",
    model_kwargs = {'device':'cpu'}
)
vector_db = FAISS.load_local(r"D:\Anurag-Agri-Bot\datas\faiss\custum_website/", hfembeddings,allow_dangerous_deserialization=True)
retrieval_qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type="stuff",
                                retriever=vector_db.as_retriever(search_kwargs={'k': 1}),
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": custom_prompt_template}
                            )



from speech import recognize_speech , text_to_speech
flag=True
while flag:
    text_to_speech("welcome to Agri-Bot, press one to know more and 0 to skip to free style")
    x = int(input("enter the number : "))
    if x==1:
        text_to_speech("press one for the news , press two for the others recent post , press three for the store information , press four for the general queries , press five for the getting financial information , press zero to end the conversation , bye")
        x = int(input("enter the number : "))
        if x==1:
            text_to_speech("press one for getting hot news , press two for getting crop price , press three for getting free style")
            y = int(input("enter the number : "))
            if y==1:
                user_input = "what is top news and treading news in the agricultural sector ? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if y==2:
                user_input = "what is crops prices for various crops in different states ? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if y==3:
                user_input = "what is news going on ? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
        elif x==2:
            text_to_speech("press one for checking personal message , press two for trending messages , press three for getting free style")
            y = int(input("enter the number : "))
            if y==1:
                user_input ="Which is the recent message in the inbox ? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if y==2:
                user_input = "What is the trending news ? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if y==3:
                user_input = "What is the recenet news about crop prices in market ? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
        elif x==3:
            text_to_speech("press one for getting crop price , press two for getting machinery price, press three for getting free style")
            x = int(input("enter the number : "))
            if y==1:
                user_input = "What is the price of jowar? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if y==2:
                user_input = "What is the price of an tractor ?  "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if y==3:
                user_input = "Suggest me an fertilizer which has ammonium in it"
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
        elif x==4:
            text_to_speech("tell your query")
            user_input = "What are the insurance plans available for cattles ? "
            prompt = {'query': user_input}
            model_out = retrieval_qa_chain(prompt)
            answer = model_out['result']
            print(answer)
            text_to_speech(answer)
        elif x==5:
            text_to_speech("press one for loan details , press two for insurance details , press three for checking the profit")
            y = int(input("enter the number : "))
            if y==1:
                user_input = "Suggest me a loan plan that has duration of 5 years ? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if y==2:
                user_input = "What are the insurance available for crops ? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if y==3:
                user_input = "What is the current profit ? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
        elif x==0:
            text_to_speech("end of convo")
            break
        else:
            text_to_speech("choose the correct option")      
    else:
        text_to_speech("start exploring the software")
        first_input = int(input("enter first input : "))
        if first_input==1 or 2 or 3 or 5 :
            second_input = int(input("enter second input : "))
            if first_input==1 and second_input==1:
                user_input = " what is top news and treading news in the agricultural sector ?"
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if first_input==1 and second_input==2:
                user_input = "what is crops prices for various crops in different states ? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if first_input==1 and second_input==3:
                user_input = "what is news going on ?"
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if first_input==2 and second_input==1:
                user_input = "Which is the recent message in the inbox ?  "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer) 
            if first_input==2 and second_input==2:
                user_input = "What is the trending news ?"
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if first_input==2 and  second_input==3:
                user_input = "What is the recenet news about crop prices in market ?  "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if first_input==3 and second_input==1:
                user_input = "What is the price of jowar?  "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if first_input==3 and second_input==2:
                user_input = "What is the price of an tractor ? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if first_input==3 and second_input==3:
                user_input = "Suggest me an fertilizer which has ammonium in it "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if first_input==5 and second_input==1:
                user_input = "Suggest me a loan plan that has duration of 5 years ?"
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if first_input==5 and second_input==2:
                user_input = "What are the insurance available for crops ? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
            if first_input==5 and second_input==3:
                user_input = "What is the current profit ? "
                prompt = {'query': user_input}
                model_out = retrieval_qa_chain(prompt)
                answer = model_out['result']
                print(answer)
                text_to_speech(answer)
        elif first_input==4:
            user_input = "What are the insurance plans available for cattles ? "
            prompt = {'query': user_input}
            model_out = retrieval_qa_chain(prompt)
            answer = model_out['result']
            print(answer)
            text_to_speech(answer)
        elif first_input==0:
            break
        else:
            text_to_speech("Go again")
