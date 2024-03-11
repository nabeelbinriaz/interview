from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
import os
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import json
import tempfile
import ast
import requests

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
openai_api_key = os.environ.get("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = openai_api_key
@app.post("/upload-cv/")
async def process_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    # Extract text from PDF
    detected_text = ""
    with open(temp_file_path, "rb") as pdf_file_obj:
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        num_pages = len(pdf_reader.pages)
        for page_num in range(0,num_pages):
            page_obj = pdf_reader.pages[page_num]
            detected_text += page_obj.extract_text() + "\n\n"

    with open("cv.txt","w",encoding="utf-8") as f:
        f.write(detected_text)      

@app.post("/interview/")
async def summarzie_audio_file(text: str = Form(...)):
    print("here......")
    # with open("cv.txt","rb") as f:
    #     cv=f.read()
    with open("chat.txt", "rb") as files:
        loaded_data = files.read().decode('utf-8')  # Decoding the binary data to string
        print("leaded_data: ", loaded_data)
    try:
        mes_dict = ast.literal_eval(loaded_data)
    except:
        mes_dict=[]
    print(mes_dict)

    messages=messages_from_dict(mes_dict)
    print("here 2......")
    retrieved_chat_history = ChatMessageHistory(messages=messages)
    print("history : ",retrieved_chat_history)
    temp = "You are an interviewer, designed to interview the Human, based on the provided 'Candidate information'. You should ask technical questions related to the candidate's field one by one and based on the 'Chat History' predict what the next question should be. When the Human say 'hello', you should say 'hello there, thankyou for taking your time for inteviewing with us, Please introduce yourself' and then wait for Human answer",f"\n\nChat History:\n\n{retrieved_chat_history}\n\nConversation:\nHuman: {text}\nAI: (Your next question based on previous response..)"
    prompt = PromptTemplate(
        input_variables=["history","input"], template=temp
    )
    llm = OpenAI(model='gpt-3.5-turbo-instruct',
                temperature=0, 
                max_tokens = 256)
    memory = ConversationBufferMemory(chat_memory=retrieved_chat_history)
    conversation = ConversationChain(
        llm=llm, 
        verbose=True, 
        memory=memory,
        prompt=prompt
    )
    output=conversation.predict(input=text)
    with open("chat.txt","w",encoding="utf-8") as ff:
        ff.write(f"{messages_to_dict(conversation.memory.chat_memory)}\n")
    return output

@app.post("/manual/")
async def detail(text: str = Form(...),name: str = Form(...),desired_job: str = Form(...),experience: str = Form(...),job_description: str = Form(...)):
    ci=f"Name: {name}, Desired Job:{desired_job}, experience: {experience}, Job Description: {job_description}"
    print(ci)
    print("API HITT.......")
    with open("chat.txt", "rb") as files:
        loaded_data = files.read().decode('utf-8')  # Decoding the binary data to string
    try:
        mes_dict = ast.literal_eval(loaded_data)
    except:
        mes_dict=[]
    print(mes_dict)

    messages=messages_from_dict(mes_dict)
    retrieved_chat_history = ChatMessageHistory(messages=messages)
    template = "You are an interviewer, designed to interview the Human, based on the provided 'Candidate information'. You should ask technical questions related to the candidate's field one by one and based on the 'Chat History' and  predict what the next question should be . When the Human say 'hello', JUST respond with 'hello there, thankyou for taking your time for inteviewing with us, Please introduce yourself'\n Donot write anything else and then wait for Human answer"+f"\n\nCandidate Information:\n\n{ci}"+"\n\nChat History:\n\n{history}\n\nConversation:\nHuman: {input}\nAI:"
    prompt = PromptTemplate(
        input_variables=["history","input"], template=template
    )
    llm = OpenAI(model='gpt-3.5-turbo-instruct',
                temperature=0, 
                max_tokens = 256)
    memory = ConversationBufferMemory(chat_memory=retrieved_chat_history)
    conversation = ConversationChain(
        llm=llm, 
        verbose=True, 
        memory=memory,
        prompt=prompt
    )
    a=conversation.predict(input=text)
    with open("chat.txt","w",encoding="utf-8") as ff:
        ff.write(f"{messages_to_dict(conversation.memory.chat_memory.messages)}\n")
    return a 

@app.post("/end/")
async def endd():
    with open("chat.txt","w",encoding="utf-8") as ff:
        pass

@app.post("/analysis/")
async def analysiss():
    with open("chat.txt","rb") as ff:
        chat=ff.read().decode('utf-8')

    URL = "https://api.openai.com/v1/chat/completions"

    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role":"system","content":"You are an HR professional reviewing candidate based on the interview chat of the candidate with an AI model\n\n ALways give your answer in follwoing format\nOverallDecision: (0-100 score based on the interview performance)\n\nOverallSentiment:0-100.\nAreasOfStrength:\nAreasToImprove:"},{"role": "user", "content": f"Based on this chat:\n{chat}\n Give analysis"}],
    "temperature" : 0.1,
    "top_p":1.0,
    "n" : 1,
    "stream": False,
    "presence_penalty":0,
    "frequency_penalty":0,
    }

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}"
    }

    response = requests.post(URL, headers=headers, json=payload, stream=False)
    res=response.json()
    content = res['choices'][0]['message']['content']
    parts = content.split('\n\n')
    separated_data = []
    for item in parts:
        # Splitting the string into two parts: before and after the first colon or newline
        parts = item.split(":", 1) if ":" in item else item.split("\n", 1)
        if len(parts) == 2:
            # If there is a colon, split into heading and content
            separated_data.append({"heading": parts[0], "content": parts[1].strip()})
        else:
            # If there is no colon, consider the whole item as a heading with no content
            separated_data.append({"heading": parts[0], "content": ""})
    heading_content_dict = {item["heading"]: item["content"] for item in separated_data}
    return heading_content_dict

@app.post("/criteria/")
async def criteriaaa():
    with open("chat.txt","rb") as ff:
        chat=ff.read().decode('utf-8')

    URL = "https://api.openai.com/v1/chat/completions"

    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role":"system","content":"You are an HR professional who give scores reviewing candidate based on the interview chat of the candidate with an AI model\n\n ALways give your answer in follwoing format\nSelfIntroduction:0-100%\nTeamworkAndCollaboration:0-100%\nProblemSolvingSkills:0-100%\nAdaptibility:0-100%,\nCommunication:0-100%"},{"role": "user", "content": f"Based on this chat:\n{chat}\n Give analysis"}],
    "temperature" : 0.1,
    "top_p":1.0,
    "n" : 1,
    "stream": False,
    "presence_penalty":0,
    "frequency_penalty":0,
    }

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}"
    }

    response = requests.post(URL, headers=headers, json=payload, stream=False)
    res=response.json()
    content = res['choices'][0]['message']['content']
    performance_dict = dict(line.split(': ') for line in content.split('\n'))
    performance_dict = {key: int(value.replace('%', '')) for key, value in performance_dict.items()}
    return performance_dict