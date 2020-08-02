from flask import Flask, render_template
from flask import jsonify
from flask import request
from keras.models import load_model
import pickle
import sklearn
import numpy as np
import random
from collections import OrderedDict
import json

app = Flask(__name__) 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

###########################################################################
@app.route('/', methods=['GET', 'POST'])
def basic():
    return render_template("index.html")

###########################################################################
@app.route('/pump', methods=['GET', 'POST'])
def pump():
    parameters = OrderedDict()
    flowRate = round(random.uniform(115, 151), 3)
    power = random.randint(27, 33)
    h = round(random.uniform(55, 60), 3)
    eff_calc = round(0.0028*(round((h*flowRate)/power, 3)), 3)
    eff_calc = eff_calc * 100
    eff_calc = round(eff_calc, 3);
    fname = 'ML-models/pump_efficiency.sav'
    loaded_model = pickle.load(open(fname, 'rb'))
    eff_pred = loaded_model.predict([[flowRate, power, h]])
    eff_pred = round(eff_pred[0][0], 3)
    if eff_pred < 70 :
        label = 0
    else:
        label = 1 
    fname = 'ML-models/flow-rate.sav'
    loaded_model = pickle.load(open(fname, 'rb'))
    flowRatePred = loaded_model.predict([[power, h, eff_pred]]) 
    flowRatePred = round(flowRatePred[0][0], 3)    
    parameters = {
        "power": power,
        "h": h,
        "flowRate": flowRate,
        "eff_calc": eff_calc,
        "eff_pred": eff_pred,
        "label": label,
        "flowRatePred": flowRatePred
    }
    return parameters

###########################################################################
@app.route('/util', methods=['GET', 'POST'])
def util():

    lmtd = round(random.uniform(6.028, 79.965), 3)
    u = round(random.uniform(0.151, 0.197), 3)
    inletTempHot = round(random.uniform(80, 124), 3)
    outletTempHot = round(random.uniform(30.302, 79.148), 3)
    inletTempCold = round(random.uniform(5,49), 3)
    outletTempCold = round(random.uniform(23.241, 76.537), 3)
    area = 2.112
    qIdeal = round((u * area * lmtd), 3)
    rand1 = round(random.uniform(0.55, 0.95), 3)
    qaNormal = round((rand1*qIdeal), 3)
    rand2 = round(random.uniform(0.30, 0.53), 3)
    qaMaintenance = round((rand2*qIdeal), 3)
    rand3 = round(random.uniform(0.10, 0.30), 3)
    qaAccidental = round((rand3*qIdeal), 3)
    filename1 = 'ML-models/finalized_model_normal.sav'
    filename2 = 'ML-models/finalized_model_maintenance.sav'
    filename3 = 'ML-models/finalized_model_accidental.sav'
    filename4 = 'ML-models/normal_temp.sav'
    filename5 = 'ML-models/maintenance_temp.sav'
    filename6 = 'ML-models/accidental_temp.sav'
    loaded_model = pickle.load(open(filename1, 'rb'))
    EffNormal = loaded_model.predict([[qaNormal,qIdeal]])
    loaded_model = pickle.load(open(filename2, 'rb'))
    EffMaintenance = loaded_model.predict([[qaMaintenance,qIdeal]])
    loaded_model = pickle.load(open(filename3, 'rb'))
    EffAccidental = loaded_model.predict([[qaAccidental,qIdeal]])
    randEff = ['EffNormal', 'EffMaintenance', 'EffAccidental']
    nn = random.randint(0,2)
    if nn==0:
        loaded_model = pickle.load(open(filename4, 'rb'))
        tempNormal = loaded_model.predict(np.array([[u,lmtd,EffNormal]],dtype=object))
        tempHot = tempNormal[0][0]
        tempCold = tempNormal[0][1]
    elif nn==1:
        loaded_model = pickle.load(open(filename5, 'rb'))
        tempMaintenance = loaded_model.predict(np.array([[u,lmtd,EffMaintenance]],dtype=object))
        tempHot = tempMaintenance[0][0]
        tempCold = tempMaintenance[0][1]
    else:
        loaded_model = pickle.load(open(filename6, 'rb'))
        tempAccidental = loaded_model.predict(np.array([[u,lmtd,EffAccidental]],dtype=object))
        tempHot = tempAccidental[0][0]
        tempCold = tempAccidental[0][1]

    parameters = {
        "area":21.12,
        "lmtd": lmtd,
        "u": u,
        "qIdeal": qIdeal,
        "qaNormal": qaNormal,
        "qaMaintenance": qaMaintenance,
        "qaAccidental": qaAccidental,
        "EffNormal": round(EffNormal[0], 3),
        "EffMaintenance": round(EffMaintenance[0], 3),
        "EffAccidental": round(EffAccidental[0], 3),
        "randEff": randEff[nn],
        "inletTempHot": inletTempHot,
        "outletTempHot": outletTempHot,
        "inletTempCold": inletTempCold,
        "outletTempCold": outletTempCold,
        "outletTempHotPred":round(tempHot,3),
        "outletTempColdPred":round(tempCold,3),
    }
    return parameters

#########################################################
@app.route('/he', methods=['GET', 'POST'])
def he():

    lmtd = round(random.uniform(208.49, 249.73), 3)
    u = round(random.uniform(0.344, 0.908), 3)
    inletTempHot = round(random.randint(340, 360), 3)
    outletTempHot = round(random.randint(220, 245), 3)
    inletTempCold = round(random.randint(30, 45), 3)
    outletTempCold = round(random.randint(70,95), 3)
    area = 21.12
    qIdeal = round((u * area * lmtd), 3)
    rand1 = round(random.uniform(0.15, 0.92), 3)
    qaNormal = round((rand1*qIdeal), 3)

    model  = load_model('ML-models/ANN-version-3.h5',compile=False)
    
    ar = [[inletTempHot, inletTempCold, outletTempHot, outletTempCold, u, lmtd, qaNormal, qIdeal]]
    efficiency = model.predict(ar)
    efficiency = round(efficiency[0][0], 3)
    print(efficiency)
    if efficiency<30:
        randEff = 'EffAccidental'
    elif efficiency>=30 and efficiency<=60:
        randEff = 'EffMaintenance'
    else:
        randEff = 'EffNormal'  

    model1  = load_model('ML-models/Temp_heat_out.h5',compile=False)  
    ar = [[inletTempHot, inletTempCold, u, lmtd, qaNormal, qIdeal]]  
    outletTempHotPred = model1.predict(ar)
    outletTempHotPred = round(outletTempHotPred[0][0], 3)

    model2  = load_model('ML-models/Temp_cold_out.h5',compile=False)  
    ar = [[inletTempHot, inletTempCold, u, lmtd, qaNormal, qIdeal]]  
    outletTempColdPred = model2.predict(ar)
    outletTempColdPred = round(outletTempColdPred[0][0], 3)
    parameters = {
        "area":str(21.12),
        "lmtd": str(lmtd),
        "u": str(u),
        "qIdeal": str(qIdeal),
        "qaNormal": str(qaNormal),
        "efficiency": str(efficiency),
        "randEff": str(randEff),
        "inletTempHot": str(inletTempHot),
        "inletTempCold": str(inletTempCold),
        "outletTempHot": str(outletTempHot),
        "outletTempCold": str(outletTempCold),
        "outletTempHotPred": str(outletTempHotPred),
        "outletTempColdPred":str(outletTempColdPred)
    }
    return parameters  

if __name__ == '__main__': 
    app.run()    