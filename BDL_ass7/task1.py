from typing import Union
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model as keras_load_model
import sys, uvicorn
from time import time
import psutil

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram



IMAGEDIR = "images/"
app = FastAPI()

num_requests = Counter('num_requests', 'Number of requests received', ['method', 'endpoint', 'ip_address'])
processing_time_per_char = Gauge('processing_time_per_char', 'Processing time per character in microseconds', ['method', 'endpoint'])
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
network_io_sent = Counter('network_io_sent_bytes_total', 'Total number of bytes sent via network')
network_io_received = Counter('network_io_received_bytes_total', 'Total number of bytes received via network')


Instrumentator().instrument(app).expose(app)

#model_path = sys.argv[1]
model_path = "C:\\Users\\91990\Downloads\\FastAPI\\mnist-epoch-10.hdf5"

def load_model(model_path):
    model = keras_load_model(model_path)
    return model

def format_image(image):
    # Open the image using PIL
    # Convert the image to grayscale
    image = image.convert("L")
    #image = ImageOps.invert(image)
    # Resize the image to 28x28
    image = image.resize((28, 28))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Flatten the image array to a 1D array of 784 elements
    flattened_image = image_array.reshape(1, 784)
    # Normalize the pixel values to the range [0, 1]
    normalized_image = flattened_image / 255.0
    return normalized_image

def predict(img):
    #print(x.shape)
    x = format_image(img)
    model = load_model(model_path)
    prediction = model.predict(x)
    digit = np.argmax(prediction)
    return str(digit)


@app.post("/upload/")
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    num_requests.labels(method="POST", endpoint="/upload", ip_address=request.client.host).inc()

    start_time = time()
    try:
        file.filename = f"{uuid.uuid4()}.jpg"
        contents = await file.read()
    
        #save the file
        with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
            f.write(contents)
        
        img = Image.open(f'{IMAGEDIR}{file.filename}')

        
        start_time = time()
        ans = predict(img)

        os.remove(f'{IMAGEDIR}{file.filename}')

        end_time = time()
        processing_duration = (end_time - start_time) * 1000  # Convert to milliseconds
        input_length = 784
        processing_time_per_char_value = (processing_duration / input_length) * 1000  # Convert to microseconds per character
        processing_time_per_char.labels(method="POST", endpoint="/upload").set(processing_time_per_char_value)
        
        # System Metrics Update
        memory_usage.set(psutil.virtual_memory().used)
        cpu_usage.set(psutil.cpu_percent())
        net_io = psutil.net_io_counters()
        network_io_received.inc(net_io.bytes_recv)
        network_io_sent.inc(net_io.bytes_sent)
        return {"digit": ans}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app)