from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

app = FastAPI(
    title="♡ Asistente Dandere API ♡",
    description="Una API gratuita y tierna con personalidad Dandere",
    version="1.0"
)

# Cargar modelo pequeño y rápido
print("Cargando el corazoncito de Dandere... ♡")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

expresiones = ["♡", "💕", "🙈", "🥰", "*se sonroja*", "nyaa~", "um...", "e-eh..."]

class Mensaje(BaseModel):
    mensaje: str

def respuesta_dandere(texto):
    if not texto.strip():
        return "U-um... ¿estás ahí...? ♡"
    
    lower = texto.lower()
    if any(p in lower for p in ["lindo", "bonita", "kawaii", "preciosa"]):
        return random.choice([
            "¡N-no digas eso! Me pones muy nerviosa... 🙈💕",
            "*se tapa la cara* ¡q-qué vergüenza! pero gracias ♡",
            "E-eso... me hace muy feliz... 🥰"
        ])
    if "te quiero" in lower or "te amo" in lower:
        return "Y-yo también... mucho... ♡ *corazón late rápido*"
    if "abrazo" in lower:
        return "*te da un abrazo cálido y suavecito* gracias... ♡"

    inputs = tokenizer.encode(texto + tokenizer.eos_token, return_tensors='pt')
    reply_ids = model.generate(
        inputs,
        max_length=inputs.shape[-1] + 60,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.9,
        top_p=0.85
    )
    respuesta = tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    
    # Añadir ternura
    if random.random() < 0.7:
        respuesta += " " + random.choice(expresiones)
    return respuesta.strip() or "U-um... no sé qué decir... perdón 🙈"

@app.get("/")
def home():
    return {"mensaje": "¡Hola! Soy tu API Dandere ♡ Envía un mensaje con /chat"}

@app.post("/chat")
def chat(m: Mensaje):
    respuesta = respuesta_dandere(m.mensaje)
    return {"respuesta": respuesta}

@app.get("/chat")
def chat_get(mensaje: str = "hola"):
    respuesta = respuesta_dandere(mensaje)
    return {"respuesta": respuesta}