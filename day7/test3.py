import tensorflow as tf
import numpy as np

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
# 데이터 사전화
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑']]


# 입력단어와 출력단어를 한글자씩 떼어낸 후 배열로 만들고, 원-핫 인코딩 형식으로 만드는 함수
# 디코더에 입력이 시작되었다는 심볼 : S
# 디코더의 출력이 끝났음을 알려주는 심볼 : E
# 빈 데이터를 채울때 의미없는 값임을 알려주는 심볼 : P
def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        # 손실함수로 sparse_softmax_cross_entropy_with_logits 를 사용할 계획이므로 원-핫 인코딩을 하지 않음
        target_batch.append(target)

    return input_batch, output_batch, target_batch


# 하이퍼 파라미터
learning_rate = 0.01
n_hidden = 128
total_epoch = 100

n_class = n_input = dic_len

# 인코더와 디코더의 입력값 형식은 [batch size, time steps, input_size]
# 디코더의 출력값 형식은 [batch size, time steps]
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])

# RNN Cell 구성
# encoder 셀 구성
with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)

# decoder 셀 구성
with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    # encoder에서 계산 후 decoder로 전파 되어야 한다는
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_states, dtype=tf.float32)

model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=model))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    input_batch, output_batch, target_batch = make_batch(seq_data)

    for epoch in range(total_epoch):
        _, loss = sess.run([train_op, cost],
                           feed_dict={enc_input: input_batch, dec_input: output_batch, targets: target_batch})

        print('epoch:', '%04d' % (epoch + 1), 'cost:', '{:6f}'.format(loss))

    print('최적화 완료')


    def translate(word):
        seq_data = [word, 'P' * len(word)]

        input_batch, output_batch, target_batch = make_batch([seq_data])

        prediction = tf.argmax(model, 2)
        result = sess.run(prediction,
                          feed_dict={enc_input: input_batch, dec_input: output_batch, targets: target_batch})

        decoded = [char_arr[i] for i in result[0]]
        end = decoded.index('E')
        translated = ''.join(decoded[:end])
        return translated


    print("번역 테스트")

    print('word:', translate('word'))
    print('wood:', translate('wood'))
    print('love:', translate('love'))
    print('loev:', translate('loev'))
    print('abcd:', translate('abcd'))
    print('kiss:', translate('kiss'))
