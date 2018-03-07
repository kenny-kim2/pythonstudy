import os.path
import tensorflow
import json
from konlpy.tag import Twitter

class DeepCNModule:
    def __init__(self):
        self.step = 1000
        self.batch_size = 100
        self.learning_rate = 0.01

        self.twitter = Twitter()

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

        self.vocab = {}
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
                self.tokenize_word_data(model, 'model')
                self.tokenize_word_data(prodname, 'name')
                self.add_vocab(lcatecode, 'cate')
                self.add_vocab(price, 'price')
                self.add_vocab(cmpny_cate, 'cmpnycate')
                self.add_vocab(img_code, 'img')
        self.show_dict_status()

        self.save_vocab()

    def tokenize_word_data(self, data, param):
        result = {}
        for data in self.twitter.pos(data, norm=True, stem=True):
            if data[1] == 'Foreign':
                continue
            if data[1] == 'Punctuation':
                continue
            if data[1] == 'Josa':
                continue
            self.add_vocab(data[0], param)

    def add_vocab(self, data, param):
        if data not in self.vocab[param].keys():
            self.vocab[param][data] = len(self.vocab[param])

    def save_vocab(self):
        for data in self.vocab_path_dict:
            with open(self.vocab_path+'/'+self.vocab_path_dict[data], 'w') as f:
                f.write(json.dumps(self.vocab[data]))


    def read_vocab(self, path):
        vocab = {}
        with open(self.vocab_path+"/"+path, 'r') as f:
            vocab = json.loads(f.read())

        return vocab

    def show_dict_status(self):
        print('name vocab = ', str(len(self.vocab['name'])))
        print('maker vocab = ', str(len(self.vocab['maker'])))
        print('model vocab = ', str(len(self.vocab['model'])))
        print('cate vocab = ', str(len(self.vocab['cate'])))
        print('price vocab = ', str(len(self.vocab['price'])))
        print('cmpnycate vocab = ', str(len(self.vocab['cmpnycate'])))
        print('img vocab = ', str(len(self.vocab['img'])))

    def make_model(self):


        pass

    def train(self):
        pass

    def predict(self):
        pass

