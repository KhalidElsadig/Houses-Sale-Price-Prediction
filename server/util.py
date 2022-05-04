import  json 
from joblib import load
import numpy as np
from numpy.lib.financial import nper

__data_columns = [] 
__column =[]
__model = []

def get_predicted_price(yearremodadd,onestflrsf,twondflrsf,grlivarea,bedroomabvgr,kitchenabvgr,totrmsabvgrd,garageyrblt,garagecars,mosold,age,home_quality,area,numbath,porch_area,haspool,hasfireplace,exterqual,kitchenqual,bsmtqual,garagefinish,foundation):
    x=np.zeros(len(__data_columns))
    x[0]=yearremodadd
    x[1]=onestflrsf
    x[2]=twondflrsf
    x[3]=grlivarea
    x[4]=bedroomabvgr
    x[5]= kitchenabvgr
    x[6]= totrmsabvgrd
    x[7]=  garageyrblt
    x[8]= garagecars
    x[9]= mosold
    x[10]= age
    x[11]= home_quality
    x[12]= area
    x[13]=numbath
    x[14]=porch_area
    x[15]= haspool
    x[16]=hasfireplace
    x[17]=exterqual
    x[18]=kitchenqual
    x[19]=bsmtqual
    x[20]=garagefinish
    x[21]=foundation



    return round(__model.predict([x])[0],0)
    

def load_saved_model_data():
    print('loding model data ')
    global __data_columns
    global __column

    with open(r'C:\Users\KHALID\vs code\data science projects\Sale House predection using machine learning\model\columns.json','r') as f :
        __data_columns = json.load(f)['data_columns']
        __column = __data_columns
        

    with open (r'C:\Users\KHALID\vs code\data science projects\Sale House predection using machine learning\model\Decision Tree model.pkl','r') as f:
            __model= load(f)
            print('loading model data is done')  
            return __column
variables = print(','.join(c for c in (__column) if c not in "'"))            
if __name__ == 'main':
    load_saved_model_data()
    print(get_predicted_price(variables))

    