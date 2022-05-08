#import Libaries 
from typing import Optional
from fastapi import FastAPI, Form,Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import  json 
from joblib import load
import numpy as np
import uvicorn
from scipy.special import inv_boxcox
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
## load Model 
__data_columns = [] 
__model = []
def get_predicted_price(num_bath,porch_area,yearremodadd,firstfloorarea,home_quality,Age, garagecars,area,ExterQual,grlivarea):
    try:
        loc_index = __data_columns.index()
    except :
            loc_index=-1
    x=np.zeros(len(__data_columns))
    x[0]= num_bath
    x[1]= porch_area
    x[2]= yearremodadd
    x[3]= firstfloorarea
    x[4]= home_quality
    x[5]= Age
    x[6]= garagecars
    x[7]= area
    x[8]= ExterQual
    x[9]= grlivarea



    return int(round(inv_boxcox(__model.predict([x])[0],-0.20003295224564416),0))


def load_saved_model_data():
    print('loading model data ')
    global __data_columns
    global __model
    pathcolumns= "columns.json"
    with open(pathcolumns,'r') as f :
        __data_columns = json.load(f)['data_columns']
        print('columns:',__data_columns)
    pathmodel= "Random forest model.pkl"

    with open (pathmodel,'rb') as f:
            __model= load(f)
            print('loading model data is done')

    variables = print(','.join(c for c in (__data_columns) if c not in "'"))


# Build web app
load_saved_model_data()
app = FastAPI()
templates = Jinja2Templates("templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get('/',response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/predicted_price', response_class=HTMLResponse)
async def predict_price(request: Request,
num_bath:float= Form(...),
porch_area:float= Form(...),
yearremodadd:float= Form(...),
firstfloorarea:float= Form(...),
home_quality:float= Form(...),
Age:float= Form(...),
garagecars:float= Form(...),
area:float= Form(...),
ExterQual:float= Form(...),
grlivarea:float= Form(...),
):
    response =get_predicted_price(num_bath,porch_area,yearremodadd,firstfloorarea,home_quality,Age, garagecars,area,ExterQual,grlivarea)
    return templates.TemplateResponse("result.html", {"request": request,"id": response})
#Run web app
if __name__ == "__main__":
    uvicorn.run(app,host='127.0.0.1',port=8000)
