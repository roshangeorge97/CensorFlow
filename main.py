import threading
import time
import datetime
import uuid
import uvicorn
import gradio as gr
from fastapi import FastAPI, Request, Header, HTTPException
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
import os
import google.generativeai as genai


# Constants
load_dotenv()
API_VERSION = "v1"
SERVICE_NAME = "ContentModerationService"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
SUPPORTED_METHODS = ["moderate_text", "moderate_batch", "suggest_text"]

# Model Metadata
MODEL_INFO = {
    "name": "KoalaAI/Text-Moderation",
    "description": "Text classification model based on Deberta-v3 that predicts whether a text contains offensive content.",
    "categories": {
        "S": "Sexual content",
        "H": "Hate speech",
        "V": "Violence",
        "HR": "Harassment",
        "SH": "Self-harm",
        "S3": "Sexual content involving minors",
        "H2": "Hate speech with threats",
        "V2": "Graphic violence",
        "OK": "Not offensive"
    },
    "language": "English",
    "license": "CodeML OpenRAIL-M 0.1",
    "performance": {
        "accuracy": 0.749,
        "macro_f1": 0.326,
        "weighted_f1": 0.703
    },
    "ethical_considerations": [
        "May reinforce or amplify existing biases or stereotypes.",
        "Consider purpose, context, and impact of using this model.",
        "Respect privacy and consent of data subjects.",
        "Adhere to relevant laws and regulations."
    ]
}

# Initialize FastAPI app
app = FastAPI()

# Load Hugging Face models
tokenizer = AutoTokenizer.from_pretrained(MODEL_INFO["name"])
model = AutoModelForSequenceClassification.from_pretrained(MODEL_INFO["name"])

# Load Gemini API
genai.configure(api_key=os.environ['API_KEY'])

# Task management
task_status = {}
task_results = {}

# Pydantic models
class ModerateTextRequest(BaseModel):
    text: str

class ModerateBatchRequest(BaseModel):
    texts: List[str]

class SuggestTextRequest(BaseModel):
    text: str
    moderation_result: Dict[str, float]

class TaskRequest(BaseModel):
    taskId: str

# Helper functions
def moderate_text(text: str) -> Dict[str, float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    return {label: prob.item() for label, prob in zip(MODEL_INFO["categories"].keys(), probabilities)}

def suggest_text(text: str, moderation_result: Dict[str, float]) -> str:
    # Construct the prompt for Gemini API
    prompt = f"""Given the following text and its moderation results, suggest an improved version that addresses any potential issues while maintaining the original intent:

Text: {text}

Moderation Results:
{' '.join([f'{MODEL_INFO["categories"][k]}: {v:.2%}' for k, v in moderation_result.items()])}

Suggested Improvement:"""
    
    # Use Gemini API to generate content
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Assuming 'generate_content' is the incorrect method,
        # You might need to replace it with the correct method:
        response = model.generate_content(prompt)
        suggested_text = response.text
        return suggested_text.split("Suggested Improvement:")[-1].strip()
    except Exception as e:
        print(f"Error generating content with Gemini API: {e}")
        return "Error generating suggestion"


def process_task(task_id: str, method: str, payload: dict):
    start_time = time.time()
    task_status[task_id] = "INPROGRESS"

    try:
        if method == "moderate_text":
            result = moderate_text(payload["text"])
        elif method == "moderate_batch":
            result = [moderate_text(text) for text in payload["texts"]]
        elif method == "suggest_text":
            moderation_result = moderate_text(payload["text"])
            suggestion = suggest_text(payload["text"], moderation_result)
            result = {"moderation_result": moderation_result, "suggested_text": suggestion}
        else:
            raise ValueError(f"Unsupported method: {method}")

        task_status[task_id] = "SUCCESS"
        task_results[task_id] = result
    except Exception as e:
        task_status[task_id] = "ERROR"
        task_results[task_id] = {"error": str(e)}

    process_duration = int((time.time() - start_time) * 1000)
    return process_duration

# API endpoints
@app.post("/call")
async def call_endpoint(request: Request, x_user_id: str = Header(None), x_marketplace_token: str = Header(None), x_user_role: str = Header(None)):
    start_time = time.time()
    request_data = await request.json()

    if not all([x_user_id, x_marketplace_token, x_user_role]):
        return response_template(str(uuid.uuid4()), str(uuid.uuid4()), -1, True, {}, {"status": "ERROR", "reason": "Missing required headers"})

    method = request_data.get('method')
    if not method or method not in SUPPORTED_METHODS:
        return response_template(str(uuid.uuid4()), str(uuid.uuid4()), -1, True, {}, {"status": "ERROR", "reason": "Unsupported or missing method"})

    payload = request_data.get('payload', {})
    task_id = str(uuid.uuid4())
    task_status[task_id] = "PENDING"

    threading.Thread(target=process_task, args=(task_id, method, payload)).start()

    process_duration = int((time.time() - start_time) * 1000)
    return response_template(str(uuid.uuid4()), str(uuid.uuid4()), process_duration, False, {"taskId": task_id}, {"status": "PENDING", "reason": "Task is pending"})

@app.post("/result")
async def result(request: Request, task_request: TaskRequest, x_user_id: str = Header(None), x_marketplace_token: str = Header(None), x_user_role: str = Header(None)):
    if not all([x_user_id, x_marketplace_token, x_user_role]):
        raise HTTPException(status_code=400, detail="Missing required headers")

    status = task_status.get(task_request.taskId, "ERROR")
    if status == "SUCCESS":
        result = task_results.get(task_request.taskId, {"message": "No result found"})
        return response_template(str(uuid.uuid4()), str(uuid.uuid4()), 0, True, {"taskId": task_request.taskId, "data": result}, {"status": "SUCCESS", "reason": "Task completed successfully"})
    elif status == "PENDING":
        return response_template(str(uuid.uuid4()), str(uuid.uuid4()), 0, False, {"taskId": task_request.taskId}, {"status": "PENDING", "reason": "Task is still pending"})
    elif status == "INPROGRESS":
        return response_template(str(uuid.uuid4()), str(uuid.uuid4()), 0, False, {"taskId": task_request.taskId}, {"status": "INPROGRESS", "reason": "Task is processing"})
    else:
        return response_template(str(uuid.uuid4()), str(uuid.uuid4()), 0, True, {}, {"status": "ERROR", "reason": "Task failed or error occurred"})

@app.get("/model-info")
async def get_model_info():
    return MODEL_INFO

# Response template
def response_template(request_id, trace_id, process_duration, is_response_immediate, response, error_code):
    return {
        "requestId": request_id,
        "traceId": trace_id,
        "apiVersion": API_VERSION,
        "service": SERVICE_NAME,
        "datetime": datetime.datetime.now().isoformat(),
        "processDuration": process_duration,
        "isResponseImmediate": is_response_immediate,
        "extraType": "others",
        "response": response,
        "errorCode": error_code
    }

# Gradio interface
def moderate_text_ui(text):
    response = moderate_text(text)
    return "\n".join([f"{MODEL_INFO['categories'][k]}: {v:.2%}" for k, v in response.items()])

def moderate_batch_ui(texts):
    texts_list = [text.strip() for text in texts.split('\n') if text.strip()]
    responses = [moderate_text(text) for text in texts_list]
    return "\n\n".join([f"Text: {text}\n" + "\n".join([f"{MODEL_INFO['categories'][k]}: {v:.2%}" for k, v in response.items()]) for text, response in zip(texts_list, responses)])

def suggest_text_ui(text):
    moderation_result = moderate_text(text)
    suggestion = suggest_text(text, moderation_result)
    return f"Moderation Result:\n" + "\n".join([f"{MODEL_INFO['categories'][k]}: {v:.2%}" for k, v in moderation_result.items()]) + f"\n\nSuggested Improvement:\n{suggestion}"

with gr.Blocks() as demo:
   
    with gr.Tab("Single Text Moderation"):
        text_input = gr.Textbox(label="Enter text to moderate")
        text_output = gr.Textbox(label="Moderation Result")
        text_button = gr.Button("Moderate Text")

    with gr.Tab("Batch Text Moderation"):
        batch_input = gr.Textbox(label="Enter texts to moderate (one per line)", lines=5)
        batch_output = gr.Textbox(label="Batch Moderation Results")
        batch_button = gr.Button("Moderate Batch")

    with gr.Tab("Text Suggestion"):
        suggest_input = gr.Textbox(label="Enter text for moderation and suggestion")
        suggest_output = gr.Textbox(label="Moderation Result and Suggestion")
        suggest_button = gr.Button("Moderate and Suggest")

    text_button.click(moderate_text_ui, inputs=[text_input], outputs=[text_output])
    batch_button.click(moderate_batch_ui, inputs=[batch_input], outputs=[batch_output])
    suggest_button.click(suggest_text_ui, inputs=[suggest_input], outputs=[suggest_output])

# Run both FastAPI and Gradio
if __name__ == "__main__":
    import nest_asyncio
    from fastapi.middleware.cors import CORSMiddleware
    
    nest_asyncio.apply()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=8000)