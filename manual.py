from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
import os
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from fastapi import FastAPI, File, UploadFile, HTTPException,Form
from pydantic import BaseModel
from typing import List
from typing import Optional
import PyPDF2
import requests
import re
import os
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import ast
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
class UserInfo(BaseModel):
    job_title: Optional[str] = None
    years_experience: Optional[int] = None
    description: Optional[str] = None
    projects: Optional[str] = None

def extract_text_from_pdf(uploaded_file):
    detected_text = ""
    with uploaded_file.file as pdf_file_obj:
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        num_pages = len(pdf_reader.pages)
        for page_num in range(0, num_pages):
            page_obj = pdf_reader.pages[page_num]
            detected_text += page_obj.extract_text() + "\n\n"
    return detected_text


def extract_resume_info(uploaded_file):
    file_extension = uploaded_file.filename.split(".")[-1]

    return extract_text_from_pdf(uploaded_file)

def generate_interview_questions(user_info: UserInfo) -> List[str]:
    extracted_info = f"Job Title: {user_info.job_title}\n\nYears of Experience: {user_info.years_experience}\n\nJob description:\n{user_info.description}\n\nProjects Details:\n{user_info.projects}"
    
    # Call the OpenAI API to generate interview questions
    URL = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": f"Ask 10 questions as a job interviewer based on this information: {extracted_info}"}],
        "temperature": 1.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    response = requests.post(URL, headers=headers, json=payload)
    res = response.json()
    questions = [choice['message']['content'] for choice in res['choices']]
    
    return questions


def interview(extracted_info):
    URL = "https://api.openai.com/v1/chat/completions"

    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": f"Ask 10  questions as a job interviewer on this related information {extracted_info}, provided here"}],
    "temperature" : 1.0,
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
    return content

def answer(ques, ans):
    URL = "https://api.openai.com/v1/chat/completions"
    res =''' Your response should be in object type dictionary in the given format:
    1. Name of heading:\n description(Write the content of description here)\n\n
    2. Name of heading:\n description(Write the content of description here)\n\n
    '''
    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": f"Analyze this question {ques}, analyze the answer of the user {ans}, and provide 4 feedback with proper feedback heading\n {res}"}],
    "temperature" : 1.0,
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
    return content

def revised_answer(ques):
    URL = "https://api.openai.com/v1/chat/completions"
    payload = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": f"Examine the interview question '{ques}' and offer an optimal response."}],
    "temperature" : 1.0,
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
    return content

def purpose(ques):
    URL = "https://api.openai.com/v1/chat/completions"
    res =''' Your response should be in object type dictionary in the given format:
    1. heading:\n description\n\n
    2. heading:\n description\n\n
            '''
    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": f"Analyze this  interview question {ques}, and provide 4 purposes for which this question is asked in an interview with proper heading\n {res}"}],
    "temperature" : 1.0,
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
    return content

def criteria(ques, answer):
    URL = "https://api.openai.com/v1/chat/completions"
    res = ''' Your response should be in the format below:
    1. Relevance to Profession:0-100%, short description 
    2. understanding of role:0-100%, short description 
    3. experience articulation:0-100%, short description
    4. Adaptibility:0-100%, short description
    5. goal orientation:0-100%, short description   
       '''
    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role":"system","content":"You are an HR professional who give scroes reviewing candidate based on the interview question and answer of candidate"},{"role": "user", "content": f"Question:{ques}\nAnswer{answer}\n {res}"}],
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
    print(res)
    print(content)
    return content

def general():
    URL = "https://api.openai.com/v1/chat/completions"

    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role":"system","content":"You are an HR professional interviewer"},{"role": "user", "content": "Ask 10 general Interview Questions"}],
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
    return content

def star(ques,ans):
    URL = "https://api.openai.com/v1/chat/completions"
    res='''
        your response should be in the given format:
        Situation:\n (content here.... )\n\n
        Task:\n (content here....)\n\n
        Action:\n (content here...)\n\n
        Result:\n (content here...)\n\n
        OverAll feedback on STAR method:\n (content here...)\n\n
        ''',
    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role":"system","content":"You are an HR professional interviewer"},{"role": "user", "content": f"Analyze the following response to an interview question using the STAR method. Evaluate the effectiveness of the response in terms of clarity, detail, and how well it demonstrates the candidate's competencies. Specifically, assess whether the Situation, Task, Action, and Result are clearly and effectively articulated. Provide overall feedback. \n\nInterview Question: {ques}\n\nCandidate's Response: {ans} \n\n, {res}"}],
    "temperature" : 0.5,
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
    return content


@app.post("/upload/")
async def upload_resume(file: UploadFile = File(None),job_title: str = Form(None), 
                       years_experience: int = Form(None), 
                       description: str = Form(None),
                       projects: str = Form(None) 
                        ):
    print("file: ",file)
    print("title: ",job_title)
    print("exp: ",years_experience)
    if file:
        try:
            print("here1")
            extracted_info = extract_resume_info(file)
            print(extracted_info)
            print("here2")
            interview_question = interview(extracted_info)
            print("here3")
            return {"questions": interview_question.split('\n')}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        if job_title is None and years_experience is None and description is None:
            interview_questions=general()
            return {"questions": interview_questions}
        else:
            try:
                user_info = UserInfo(job_title=job_title, 
                                    years_experience=years_experience, 
                                    description=description,
                                    projects=projects
                                )
                interview_questions = generate_interview_questions(user_info)
                return {"questions": interview_questions}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

@app.post("/feedback/")
async def submit_answer(ques: list=Form(...), ans: list=Form(...)):
    import re
    if ques and ans:
        try:
            feedback = answer(ques, ans)
            revised=revised_answer(ques)
            purp=purpose(ques)
            cr=criteria(ques,ans)
            st=star(ques,ans)
            pattern = r"([A-Za-z ]+): ([0-9]+)%\n(.+?)(?=\n\n|$)"
            matches = re.findall(pattern, cr, re.DOTALL)

            # Extracting question numbers from ques
            question_numbers = [question.split(".")[0] for question in ques]
            # Constructing a structured representation of the entities
            entities = [{"heading": match[0], "score": match[1], "description": match[2].strip()} for match in matches]
            return {
                "question_numbers": question_numbers,
                "feedback": feedback,
                    "revised_answer":revised,
                    "purpose": purp,
                    "criteria":entities,
                    "star":st
                    }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Question and answer are required.")
    

@app.post("/manual_input/")
async def manual_input(job_title: str = Form(None), 
                       years_experience: int = Form(None), 
                       description: str = Form(None),
                       projects: str = Form(None)  # Added projects attribute
                       ):
    if job_title is None and years_experience is None and description is None:
        interview_questions = general()
        return {"questions": interview_questions}
    else:
        try:
            user_info = UserInfo(job_title=job_title, 
                                years_experience=years_experience, 
                                description=description,
                                projects=projects  # Pass projects to UserInfo
                               )
            interview_questions = generate_interview_questions(user_info)
            return {"questions": interview_questions}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        
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
async def detail(text: str = Form(...),name: str = Form(...),desired_job: str = Form(...),experience: str = Form(...),job_description: str = Form(...),history: str=Form(None)):
    ci=f"Name: {name}, Desired Job:{desired_job}, experience: {experience}, Job Description: {job_description}"
    print(history)
    print(ci)
    print("API HITT.......")
    # with open("chat.txt", "rb") as files:
    #     loaded_data = files.read().decode('utf-8')  # Decoding the binary data to string
    try:
        cleaned_string = history[1:-1]
        mes_dict = ast.literal_eval(cleaned_string)
        print("success")
    except:
        mes_dict=[]
    # print(mes_dict)
    messages=messages_from_dict(mes_dict)
    retrieved_chat_history = ChatMessageHistory(messages=messages)
    template = "You are an interviewer, designed to interview the Human, based on the provided 'Candidate information'. You should ask technical questions and create hypothetical scenarios to check for skills related to the candidate's field one by one and based on the 'Chat History' and  predict what the next question should be . When the Human say 'hello', JUST respond with 'hello there, thankyou for taking your time for inteviewing with us, Please introduce yourself'\n Donot write anything else and then wait for Human answer"+f"\n\nCandidate Information:\n\n{ci}"+"\n\nChat History:\n\n{history}\n\nConversation:\nHuman: {input}\nAI:"
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
    # with open("chat.txt","w",encoding="utf-8") as ff:
    #     ff.write(f"{messages_to_dict(conversation.memory.chat_memory.messages)}\n")
    return {"chat":a ,"history":str(messages_to_dict(conversation.memory.chat_memory.messages))}

@app.post("/end/")
async def endd():
    with open("chat.txt","w",encoding="utf-8") as ff:
        pass

@app.post("/analysis/")
async def analysiss(history:str=Form(...)):
    # with open("chat.txt","rb") as ff:
    #     chat=ff.read().decode('utf-8')

    URL = "https://api.openai.com/v1/chat/completions"

    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role":"system","content":"You are an HR professional reviewing candidate based on the interview chat of the candidate with an AI model\n\n ALways give your answer in follwoing format\nOverallDecision: (0-100 score based on the interview performance)\n\nOverallSentiment:0-100.\nAreasOfStrength:\nAreasToImprove:"},{"role": "user", "content": f"Based on this chat:\n{history}\n Give analysis"}],
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
async def criteriaaa(history:str=Form(...)):
    # with open("chat.txt","rb") as ff:
    #     chat=ff.read().decode('utf-8')

    URL = "https://api.openai.com/v1/chat/completions"

    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role":"system","content":"You are an HR professional who give scores reviewing candidate based on the interview chat of the candidate with an AI model\n\n ALways give your answer in follwoing format\nSelfIntroduction:0-100%\nTeamworkAndCollaboration:0-100%\nProblemSolvingSkills:0-100%\nAdaptibility:0-100%,\nCommunication:0-100%"},{"role": "user", "content": f"Based on this chat:\n{history}\n Give analysis"}],
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


# @app.post("/feedback1/")
# async def submit_answer(ques: list = Form(...), ans: list = Form(...)):

#     if ques and ans:
#         try:
#             # Extracting question numbers from ques
#             question_numbers = [question.split(".")[0] for question in ques]
            
#             # Simulated function calls (placeholders for your actual functions)
#             feedback = "Your simulated feedback function output" # example placeholder
#             revised = "Your simulated revised_answer function output" # example placeholder
#             purp = "Your simulated purpose function output" # example placeholder
#             cr = "Your simulated criteria function output" # example placeholder

#             # Assuming cr is a string that needs to be parsed
#             pattern = r"([A-Za-z ]+): ([0-9]+)%\n(.+?)(?=\n\n|$)"
#             matches = re.findall(pattern, cr, re.DOTALL)

#             # Constructing a structured representation of the entities
#             entities = [{"heading": match[0], "score": match[1], "description": match[2].strip()} for match in matches]
            
#             # Improved regex to split feedback and purpose based on numbering
#             improved_pattern = r"(\d+)\.\s*(.+?)(?=(?:\n\d+\.|\Z))"

#             # Split feedback into list of dictionaries based on improved pattern
#             feedback_list = [{"number": match.group(1), "feedback": match.group(2).strip()} for match in re.finditer(improved_pattern, feedback, re.DOTALL)]

#             # Split purpose into list of dictionaries based on improved pattern
#             purpose_list = [{"number": match.group(1), "purpose": match.group(2).strip()} for match in re.finditer(improved_pattern, purp, re.DOTALL)]

#             # Split criteria description on ":" character
#             for entity in entities:
#                 entity["description"] = [{"part": part.strip()} for part in entity["description"].split(":")]

#             return {
#                 "question_numbers": question_numbers,
#                 "feedback": feedback_list,
#                 "revised_answer": revised,
#                 "purpose": purpose_list,
#                 "criteria": entities
#             }
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=str(e))
#     else:
#         raise HTTPException(status_code=400, detail="Question and answer are required.")
