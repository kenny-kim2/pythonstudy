import tensorflow as tf
import numpy as np

char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
           'h', 'i', 'j', 'k', 'l', 'm', 'n',
           'o', 'p', 'q', 'r', 's', 't', 'u',
           'v', 'w', 'x', 'y', 'z']

# {'a':0, 'b':1 ...}
num_dic = {n : i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 학습 데이터
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']

learning_rate = 0.01
n_hidden = 128
total_epoch = 30

n_step = 3
n_input = n_class = dic_len

# 입력값으로 단어의 처음 세글자의 인덱스를 구한 배열 생성
# 출력값으로 마지막 글자의 인덱스
# 입력값을 원-핫 인코딩 변환
def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[:-1]]
        target = num_dic[seq[-1]]
        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)

    return input_batch, target_batch

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# RNN 네트워크 구성
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
# Dropout 기법 사용
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.8)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

# 두층 이상의 RNN 네트워크를 사용하여 DeepRNN(심층 순환 신경망)을 생성
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

# 출력층 형태가 다르므로 변경
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 학습 데이터 분리
    input_batch, target_batch = make_batch(seq_data)

    for epoch in range(total_epoch):
        _, loss = sess.run([train_op, cost], feed_dict={X:input_batch, Y:target_batch})

        print('epoch:', epoch, 'loss:', loss)

    print('최적화 완료!')

    prediction = tf.cast(tf.argmax(model, 1), tf.int32)
    # Y값이 정수이므로 원-핫 인코딩으로 나온 데이터를 정수 데이터로 변경
    prediction_check = tf.equal(prediction, Y)
    accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

    input_batch, target_batch = make_batch(seq_data)

    predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: input_batch, Y: target_batch})

    predict_words = []
    for idx, val in enumerate(seq_data):
        last_char = char_arr[predict[idx]]
        predict_words.append(val[:3]+last_char)

    print('예측 결과')
    print('입력값:', [w[:3] + ' ' for w in seq_data])
    print('예측값:', predict_words)
    print('정확도:', accuracy_val)