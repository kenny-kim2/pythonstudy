from flask import Flask, render_template, request, json

# from InceptionCateModule import InceptionCateModule
# from InceptionProdModule import InceptionProdModule
from tensorflow.python.platform import gfile

app = Flask(__name__)
# inception_cate_module = InceptionCateModule()
# catemap = {}
# inception_prod_module = InceptionProdModule()

# with gfile.FastGFile('./cate.txt', 'r') as f:
#     total_data = str(f.read()).split('\n')
#     for data in total_data:
#         leafcode, cate1name, cate2name, cate3name, cate4name = data.split('\t')
#         catemap[leafcode] = cate1name+'>'+cate2name+'>'+cate3name+'>'+cate4name.replace('\r', '')
# print(catemap)


@app.route('/')
def mainPage(name=None):
    print('mainPage')
    return render_template('index.html')

@app.route('/test')
def get_prod_data():
    if request.method == 'GET':
        return render_template('test.html')

    # if request.method == 'POST':
    #
    #     test_url = request.form['image_url']
    #     print(test_url)
    #     data = []
    #     data.append(test_url)
    #
    #     result = inception_cate_module.predict(data)
    #     return_data = {}
    #     print(result)
    #     if int(result[0]) > 0:
    #         return_data['has_cate'] = True
    #         return_data['result_cate'] = result[0]
    #         return_data['input_url'] = test_url
    #         return_data['result_cate_text'] = catemap[result[0]]
    #
    #         # if '쥬얼리' in catemap[result[0]]:
    #         #     print('쥬얼리 카테고리!')
    #         #     result_prod = inception_prod_module.predict(data)
    #         #     return_data['result_prod_code'] = result_prod[0]
    #         print('쥬얼리 카테고리!')
    #         result_prod = inception_prod_module.predict(data)
    #         return_data['result_prod_code'] = result_prod[0]
    #     else:
    #         return_data['has_cate'] = False
    #         return_data['input_url'] = test_url
    #
        # return json.dumps(return_data)
    pass

@app.route('/product/detail')
def get_product_data_detail():
    pass

if __name__ == '__main__':
    app.run()
