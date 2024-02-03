from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import CustomData

application = Flask(__name__)
app = application

@app.route('/',methods=['GET','POST'])
def prediction():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(msg = str(request.form.get('sms')))
        pred_df = data.get_data_frame()
        result = data.predict(pred_df)

        return render_template('index.html',results = result[0])


if(__name__=='__main__'):
    app.run(host="0.0.0.0")