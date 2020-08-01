from flask import Flask,render_template,redirect,request
from flask_sqlalchemy import SQLAlchemy
import datetime
from flask_cors import CORS, cross_origin
app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('index2.html')

@app.route('/hev1')
def hev1():
    return render_template('heatex-v1.html')    

@app.route('/pumpv1')
def pumpv1():
    return render_template('pump-v1.html')       

@app.route('/dash')
def dash():
    return render_template('index-temp.html')

@app.route('/pump')
def pump():
    return render_template('pumpindex.html')    

@app.route('/systemsetup')
def system():
    return render_template('systemadd.html')

@app.route('/userset')
def user():
    return render_template('index-temp.html')

@app.route('/charts')
def chart():
    return render_template('chartstest.html')


if __name__ == "__main__":
    app.run(debug=True)


    