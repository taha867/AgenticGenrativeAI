from fastapi import FastAPI, Path, HTTPException, Query
import json
from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal, Optional
from fastapi.responses import JSONResponse
app = FastAPI()

class Patient(BaseModel):

    id:   Annotated[str,Field(..., description="ID of the patient", examples=["P001"])]
    name: Annotated[str,Field(..., description="Name of patient")]
    city: Annotated[str,Field(..., description="Name of City")]
    age:  Annotated[float,Field(...,gt=0, lt=120, description="Age of patient")]
    gender: Annotated[Literal["male","female","other"],Field(..., description="Gender of  of patient")]
    height: Annotated[float,Field(...,gt=0, description="Height of patient in meter's")]
    weight: Annotated[float,Field(..., gt=0, description="Weight of patient in Kg's")]

    @computed_field
    @property
    def BMI(self)->float:
        bmi= round(self.weight/(self.height**2),2)
        return bmi
    
    @computed_field
    @property
    def verdict(self)-> str:
        if self.BMI < 18.5:
            return "under weight"
        elif self.BMI < 25:
            return "Normal"
        elif self.BMI < 30:
            return "above avg"
        else:
            return "obese"


class patientUpdate(BaseModel):

    name: Annotated[Optional[str],Field(default=None)]
    city: Annotated[Optional[str],Field(default=None)]
    age:  Annotated[Optional[float],Field(gt=0, lt=120, default=None)]
    gender: Annotated[Optional[Literal["male","female","other"]],Field(default=None)]
    height: Annotated[Optional[float],Field(gt=0, default=None)]
    weight: Annotated[Optional[float],Field( gt=0, default=None)]


def loadData():
    with open("patients.json","r") as f:
        data = json.load(f)
    return data

def saveData(data):
    with open("patients.json","w") as f:
        json.dump(data, f)
    return data

@app.get("/")
def hello():
    return {"message": "Patient Management system API"}

@app.get("/about")
def about():
    return {"message": "A fully functional API to manage your patient records"}

@app.get("/view")
def view():
    data = loadData()
    return data

@app.get("/patient/{patient_id}")
def viewPatienr(patient_id: str = Path(..., description = "ID of the patient in DB", examples = "P001")):
    data =loadData()
    if patient_id in data:
        return data[patient_id]
    raise HTTPException(status_code= 404, detail= "Patient NOT found")

@app.get("/sort")
def sortPatients(sortBy: str = Query(...,description="Sort on the basis of height, weight or bmi"),
order: str = Query("asc", description="sort in ascending or descending order ")):
    
    validFields = ["height","weight","bmi"]

    if sortBy not in validFields:
        raise HTTPException(status_code= 400,detail="Invalid field select from {validFields} .")
    
    if order not in  ["asc", "desc"]:
        raise HTTPException(status_code= 400,detail="Invalid order selected between asc and desc.")
    
    data = loadData()
    sortorder = True if order == "desc" else False 
    sortedData = sorted(data.values(),key=lambda x: x.get(sortBy, 0), reverse=sortorder)

    return sortedData

@app.post("/create")
def createPatient(patient : Patient):
    data = loadData()

    if patient.id in data:
        raise HTTPException(status_code=400, detail="Patient already exist")
    
    # convert into dictionary 
    data[patient.id] = patient.model_dump(exclude=['id'])

    # save into json file 
    saveData(data)
    return JSONResponse(status_code= 201, content={"message": "Patient created sucessfully"})


@app.put("/edit/{patient_id}")
def updatePatient( patient_id: str, patient_update : patientUpdate):
    data = loadData()

    if patient_id not in data:
        raise HTTPException(status_code= 404, detail="This patient id does not exists")
    
    existing_patient_info = data[patient_id]

    # convert patient_update into dictionary

    updated_patient_info = patient_update.model_dump(exclude_unset=True)

    for key, value in updated_patient_info.items():
        existing_patient_info[key]=value

    # existing patient info -> pydantic obj -> update bmi + verdict 
    existing_patient_info["id"]=patient_id
    patient_pydantic_obj = Patient(**existing_patient_info)
    #pydamic_obj -> dict
    existing_patient_info = patient_pydantic_obj.model_dump(exclude="id")

    # ADD THID DICTIONARY TO DATA
    data[patient_id]=existing_patient_info

    saveData(data)

    return JSONResponse(status_code= 200, content={"message": "Patient updated sucessfully"})

@app.delete("/delete/{patient_id}")
def delete_patient(patient_id : str):

    data = loadData()

    if patient_id not in data:
        raise HTTPException(status_code=404 , detail=" PAtient not found")
    
    del data[patient_id]

    saveData(data)

    return JSONResponse(status_code=200, content= {"message": "patient deleted sucessfully"})