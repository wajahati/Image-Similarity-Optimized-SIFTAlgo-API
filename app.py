

# 1. Library imports
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

def compare_images(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    return len(good_matches)


def are_images_similar(inp_imgs, pro_urls):
    threshold = 50
    similar_images = []
    for img1 in inp_imgs:
        for url2 in pro_urls:
            img2 = cv2.imread(url2)
            num_good_matches = compare_images(img1, img2)
            if num_good_matches > threshold:
                similar_images.append((img1, url2))
                break
            gc.collect()
    return similar_images


@app.post('/similarityCheck')
def similarity_check(data: Similarity):
    inp_imgs = [cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR) for img in data.inpImg]
    pro_urls = data.proImg

    result = are_images_similar(inp_imgs, pro_urls)
    output_dict = {"similarity": len(result) > 0, "similar_images": result}
    return JSONResponse(content=output_dict)


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host=host, port=port)
