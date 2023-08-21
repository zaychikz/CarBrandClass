import pickle
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.code import predict_carsbrand
import requests

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# carbrandmodel = pickle.load(open(f'model/carbrandmodel.pkl','rb'))

carbrandmodel = pickle.load(open(r'..\model\carbrandmodel.pkl','rb'))

end_hog = 'http://172.17.0.1:80/api/gethog'

# end_hog = 'http://localhost:8080/api/gethog'

@app.get("/")
def root():
    return {"message": "This is my api"}

@app.post("/api/carbrand")
async def read_str(request:Request):
    item = await request.json()
    hog = requests.get(end_hog,json=item)
    res = predict_carsbrand(carbrandmodel,hog.json()['hog'])
    return res

