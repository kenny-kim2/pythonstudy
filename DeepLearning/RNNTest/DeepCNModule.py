import os.path
import tensorflow as tf
import numpy as np
import json
from konlpy.tag import Twitter

class DeepCNModule:
    def __init__(self):
        self.step = 1000
        self.batch_size = 100
        self.learning_rate = 0.01

        self.n_layers = 5
        self.n_rnn_hidden = 50
        self.n_rnn_output_size = 10
        self.n_hidden = 300

        self.epoch = 100

        self.twitter = Twitter()

        self.prod_len_max = 50

        self.data_path = './data.txt'
        if not os.path.exists(self.data_path):
            print('no data file')
            return

        self.vocab_path = './vocab'
        self.vocab_path_dict = {
            'name': 'name.txt',
            'model': 'model.txt',
            'price': 'price.txt',
            'maker': 'maker.txt',
            'cmpnycate': 'cmpnycate.txt',
            'img': 'img.txt',
            'cate': 'cate.txt'
        }

        # 사전 세팅
        self.vocab = {}
        self.make_vocab()

        self.label_size = 0

    def make_vocab(self):
        self.vocab['name'] = {}
        self.vocab['model'] = {}
        self.vocab['price'] = {}
        self.vocab['maker'] = {}
        self.vocab['cmpnycate'] = {}
        self.vocab['img'] = {}
        self.vocab['cate'] = {}
        if os.path.exists(self.vocab_path):
            for data in self.vocab_path_dict:
                if os.path.exists(self.vocab_path+"/"+self.vocab_path_dict[data]):
                    os.remove(self.vocab_path+"/"+self.vocab_path_dict[data])
            os.rmdir(self.vocab_path)

        os.makedirs(self.vocab_path)

        with open(self.data_path, 'rb') as f:
            while True:
                line = f.readline().decode('utf-8')
                if not line: break

                line = line.replace("\n", '')

                _, maker, model, prodname, lcatecode, price, cmpny_cate, img_code = line.split('\t')

                self.add_vocab(maker, 'maker')
                self.add_vocab(model, 'model')
                self.tokenize_word_data(prodname, 'name')
                self.add_vocab(lcatecode, 'cate')
                self.add_vocab(price, 'price')
                self.add_vocab(cmpny_cate, 'cmpnycate')
                self.add_vocab(img_code, 'img')
        self.show_dict_status()

        self.save_vocab()

    def tokenize_word_data(self, data, param):
        result = []
        for data in self.twitter.pos(data, norm=True, stem=True):
            if data[1] == 'Foreign':
                continue
            if data[1] == 'Punctuation':
                continue
            if data[1] == 'Josa':
                continue
            result.append(self.add_vocab(data[0], param))

        while True:
            if len(result) >= self.prod_len_max:
                break

            result.append(0)
        return result

    def add_vocab(self, data, param):
        vocab_code = 0
        if data not in self.vocab[param].keys():
            vocab_code = len(self.vocab[param]) + 1
            self.vocab[param][data] = vocab_code
        else:
            vocab_code = self.vocab[param][data]

        return vocab_code

    def save_vocab(self):
        for data in self.vocab_path_dict:
            with open(self.vocab_path+'/'+self.vocab_path_dict[data], 'w') as f:
                f.write(json.dumps(self.vocab[data]))

    def read_vocab(self, path):
        vocab = {}
        with open(self.vocab_path+"/"+path, 'r') as f:
            vocab = json.loads(f.read())

        return vocab

    def set_vocab(self):
        if os.path.exists(self.vocab_path):
            try:
                self.vocab['name'] = self.read_vocab(self.vocab_path_dict['name'])
                self.vocab['model'] = self.read_vocab(self.vocab_path_dict['model'])
                self.vocab['price'] = self.read_vocab(self.vocab_path_dict['price'])
                self.vocab['maker'] = self.read_vocab(self.vocab_path_dict['maker'])
                self.vocab['cmpnycate'] = self.read_vocab(self.vocab_path_dict['cmpnycate'])
                self.vocab['img'] = self.read_vocab(self.vocab_path_dict['img'])
                self.vocab['cate'] = self.read_vocab(self.vocab_path_dict['cate'])

                self.show_dict_status()
            except Exception as e:
                print(str(e))
                self.make_vocab()
        else:
            self.make_vocab()

    def show_dict_status(self):
        print('name vocab = ', str(len(self.vocab['name'])))
        print('maker vocab = ', str(len(self.vocab['maker'])))
        print('model vocab = ', str(len(self.vocab['model'])))
        print('cate vocab = ', str(len(self.vocab['cate'])))
        print('price vocab = ', str(len(self.vocab['price'])))
        print('cmpnycate vocab = ', str(len(self.vocab['cmpnycate'])))
        print('img vocab = ', str(len(self.vocab['img'])))

    def make_model(self):
        print('make_model')
        # input layer
        graph1 = tf.Graph()

        with graph1.as_default():
            prodname = tf.placeholder(tf.float32, [None, None, 50], name='prodname')
            maker = tf.placeholder(tf.float32, [None, None, 1], name='maker')
            model = tf.placeholder(tf.float32, [None, None, 1], name='model')
            cate = tf.placeholder(tf.float32, [None, None, 1], name='cate')
            price = tf.placeholder(tf.float32, [None, None, 1], name='price')
            cmpnycate = tf.placeholder(tf.float32, [None, None, 1], name='cmpnycate')
            img = tf.placeholder(tf.float32, [None, None, 1], name='img')

            label = tf.placeholder(tf.int32, [None], name='label')

            w = tf.Variable(tf.random_normal([285, self.label_size]))
            b = tf.Variable(tf.random_normal([self.label_size]))

            # RNN 연결
            # RNN output
            rnn_list = []

            rnn_list.append(self._getoutput_data(prodname, 'prodname', hidden_size=50))
            rnn_list.append(self._getoutput_data(maker, 'maker', hidden_size=1))
            rnn_list.append(self._getoutput_data(model, 'model', hidden_size=1))
            rnn_list.append(self._getoutput_data(cate, 'cate', hidden_size=1))
            rnn_list.append(self._getoutput_data(price, 'price', hidden_size=1))
            rnn_list.append(self._getoutput_data(cmpnycate, 'cmpnycate', hidden_size=1))
            rnn_list.append(self._getoutput_data(img, 'img', hidden_size=1))

            # concatenation layer
            concate_data = tf.concat(rnn_list, 1)

            hidden1 = tf.layers.dense(concate_data, self.n_hidden, activation=tf.nn.relu)
            hidden2 = tf.layers.dense(hidden1, self.n_hidden, activation=tf.nn.relu)
            hidden3 = tf.layers.dense(hidden2, self.n_hidden, activation=tf.nn.relu)
            hidden4 = tf.layers.dense(hidden3, self.n_hidden, activation=tf.nn.relu)

            # output layer
            output = tf.layers.dense(hidden4, self.label_size, activation=tf.nn.softmax)
            output = tf.transpose(output, [1,0,2])
            output = output[-1]
            print(output)
            model = tf.matmul(output, w) + b

            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=label))
            print('cost')
            print(cost)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            print('optimizer')
            print(optimizer)
            train_step = optimizer.minimize(cost)

            predict = tf.cast(tf.argmax(model, 1), tf.int32)

            return output, cost, train_step, graph1, predict

    def _get_data(self, outputdata, variable_name):
        with tf.variable_scope(variable_name):
            # fully connected layer
            hidden1 = tf.layers.dense(outputdata, self.n_hidden, activation=tf.nn.relu)
            hidden2 = tf.layers.dense(hidden1, self.n_hidden, activation=tf.nn.relu)
            hidden3 = tf.layers.dense(hidden2, self.n_hidden, activation=tf.nn.relu)
            hidden4 = tf.layers.dense(hidden3, self.n_hidden, activation=tf.nn.relu)

            # output layer
            output = tf.layers.dense(hidden4, self.n_rnn_output_size, activation=tf.nn.softmax)
        return output

    def _getoutput_data(self, input_data, variable_name, hidden_size):
        with tf.variable_scope(variable_name):
            outputdata, _ = tf.nn.dynamic_rnn(self._build_cells(hidden_size=hidden_size), input_data, dtype=tf.float32)

        return self._get_data(outputdata, variable_name+"layers")

    def _cell(self, output_keep_prob, hidden_size):
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=output_keep_prob)
        return rnn_cell

    def _build_cells(self, hidden_size, output_keep_prob=0.5):
        enc_cell = tf.nn.rnn_cell.MultiRNNCell([self._cell(output_keep_prob, hidden_size) for _ in range(self.n_layers)])
        return enc_cell

    def train(self):

        print('train')
        # 학습 데이터 로드
        prodname_list, maker_list, model_list, price_list, catecode_list, cmpnycate_list, imgcode_list, label_list = self.data_load(self.data_path)

        # 모델 구축
        output, cost, train_step, graph, predict = self.make_model()

        with tf.Session(graph=graph) as sess:
            # 추후 saver 추가
            sess.run(tf.global_variables_initializer())
            print(type(prodname_list))
            print(type(maker_list))
            print(type(model_list))
            print(type(price_list))
            print(type(catecode_list))
            print(type(cmpnycate_list))
            print(type(imgcode_list))
            print(type(label_list))


            for i in range(self.epoch):
                print('epoch: ', str(i))
                _, cost = sess.run([train_step, cost], feed_dict={'prodname:0': prodname_list,
                                                 'maker:0': maker_list,
                                                 'model:0': model_list,
                                                 'cate:0': catecode_list,
                                                 'price:0': price_list,
                                                 'cmpnycate:0': cmpnycate_list,
                                                 'img:0': imgcode_list,
                                                 'label:0': label_list})

                print('cost: %f' % cost)



    def data_load(self, data_path):
        prodname_list = []
        maker_list = []
        model_list = []
        price_list = []
        catecode_list = []
        cmpnycate_list = []
        imgcode_list = []

        label_list = []

        try:
            with open(data_path, 'rb') as f:
                while True:
                    line = f.readline().decode('utf-8')
                    if not line: break

                    line = line.replace("\n", '')

                    prodcode, maker, model, prodname, lcatecode, price, cmpny_cate, img_code = line.split('\t')

                    maker_list.append([[self.add_vocab(maker, 'maker')]])
                    model_list.append([[self.add_vocab(model, 'model')]])
                    prodname_list.append([self.tokenize_word_data(prodname, 'name')])
                    catecode_list.append([[self.add_vocab(lcatecode, 'cate')]])
                    price_list.append([[self.add_vocab(price, 'price')]])
                    cmpnycate_list.append([[self.add_vocab(cmpny_cate, 'cmpnycate')]])
                    imgcode_list.append([[self.add_vocab(img_code, 'img')]])

                    label_list.append(prodcode)
            self.label_dic = {n: i for i, n in enumerate(label_list)}
            return_label = [i for i, n in enumerate(label_list) ]

            self.show_dict_status()
            self.save_vocab()

        except Exception as e:
            print(str(e))
            return -1

        self.label_size = len(label_list)
        print(return_label)

        return np.array(prodname_list),\
               np.array(maker_list), \
               np.array(model_list), \
               np.array(price_list), \
               np.array(catecode_list), \
               np.array(cmpnycate_list),\
               np.array(imgcode_list),\
               np.array(return_label)


    def predict(self):
        pass

