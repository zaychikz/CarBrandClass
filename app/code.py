import pickle
import numpy as np

carBrandName = {0:'Audi', 1:'Hyundai Creta', 2:'Mahindra Scorpio',
               3:'Rolls Royce',4:'Swift', 5:'Tata Safari',
               6:'Toyota Innova'}

def predict_carsbrand(model,hog):
    brand = model.predict(np.array(hog).reshape(1,-1))
    return {'brand':carBrandName[brand[0]]}