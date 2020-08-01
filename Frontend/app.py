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

@app.route('/systemsetup')
def system():
    return render_template('systemadd.html')


if __name__ == "__main__":
    app.run(debug=True)


    