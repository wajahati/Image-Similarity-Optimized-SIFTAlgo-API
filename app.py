import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from API import Similarity
import cv2
import numpy as np
import gc
import requests
from io import BytesIO
from PIL import Image

app = FastAPI()

def download_image(url, max_size=800):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.thumbnail((max_size, max_size), Image.ANTIALIAS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def are_images_identical(url_list1, url_list2):
    identical_images = []
    for url1 in url_list1:
        img1 = download_image(url1)
        for url2 in url_list2:
            img2 = download_image(url2)
            if np.array_equal(img1, img2):
                identical_images.append((url1, url2))
            img2 = None
            gc.collect()
        img1 = None
        gc.collect()
    return identical_images

@app.post('/similarityCheck')
def profanityCheck(data:Similarity):
    inpImgs = data.inpImg
    proImgs = data.proImg
    
    result = are_images_identical(inpImgs, proImgs)
    output_dict = {"identical": len(result) > 0, "identical_images": result}
    return JSONResponse(content=output_dict)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
