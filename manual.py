from fastapi import FastAPI, File, UploadFile, HTTPException,Form
from pydantic import BaseModel
from typing import List
from typing import Optional
import PyPDF2
import requests
import re
import os
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
openai_api_key = os.environ.get("OPENAI_API_KEY")
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