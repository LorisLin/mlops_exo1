from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = FastAPI()

model_name = "t5-small" 
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

model.eval()

class TextInput(BaseModel):
    text: str

@app.post("/classify/")
async def classify(input: TextInput):
    with torch.no_grad():
        enc = tokenizer("sst2 sentence: "+ input.text, return_tensors="pt")
        decoder_input_ids = torch.tensor([tokenizer.pad_token_id]).unsqueeze(0) 
        logits = model(**enc, decoder_input_ids=decoder_input_ids)[0]
        tokens = torch.argmax(logits, dim=2)
        sentiments = tokenizer.batch_decode(tokens)
        return {"classification": sentiments}
