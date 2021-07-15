from typing import Optional
import time

from fastapi import FastAPI

import src.generate as generate
# from api_models import ResponseModel, RequestModel

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Hello World!'}

# @app.get('/samples', response_class=ResponseModel)
@app.get('/samples')
def samples_get(
    prompt: str, 
    nsamples: Optional[int] = 1, 
    model_name: Optional[str] = '124M', # Only 124M and 355M work on CPU API
    batch_size: Optional[int] = 1,
    seed: Optional[float] = None,
    length: Optional[int] = None,
    top_k: Optional[int] = 40,
    temperature: Optional[float] = 1.0
):

    start = time.time() * 1000
    output = generate.samples(prompt, model_name, seed, nsamples, batch_size, length, temperature, top_k)
    end = time.time() * 1000

    diff = end - start

    return {'prompt': prompt, 'responses': output, 'time': diff}

# @app.post('/samples', response_class=ResponseModel)
# def samples_post(data: RequestModel):

#     output = generate.samples(
#         data.prompt, 
#         data.model_name, 
#         data.seed, 
#         data.nsamples, 
#         data.batch_size, 
#         data.length, 
#         data.temperature, 
#         data.top_k
#     )

#     return {'prompt':data.prompt, 'responses':output}