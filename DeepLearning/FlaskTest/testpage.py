from flask import Flask, render_template, request, json
app = Flask(__name__)

@app.route('/')
def mainPage(name=None):
    return render_template('index.html')

@app.route('/product', methods=['POST'])
def get_product_data():
    if request.method == 'POST':
        page = int(request.form['page'])

        # 조회
        returndata = {}

        for i in range(12):
            returndata[str(i)] = {}
            returndata[str(i)]['img_url'] = 'http://i.011st.com/t/300/pd/17/7/8/6/8/7/0/zUvoE/1406786870_B.jpg'
            returndata[str(i)]['link_cnt'] = 10
            returndata[str(i)]['prod_name'] = 'test'+str(i)

        print(returndata)

        return json.dumps(returndata)

@app.route('/product/detail')
def get_product_data_detail():
    pass

@app.route('/predict', methods=['POST'])
def predict_data():
    pass

if __name__ == '__main__':
    app.run()
