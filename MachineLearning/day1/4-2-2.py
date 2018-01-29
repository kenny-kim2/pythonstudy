import pandas as pd
from sklearn import svm, metrics

# XOR 연산
xor_data = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

# 입력을 학습 전용 데이터와 테스트 전용 데이터로 분류
xor_df = pd.DataFrame(xor_data)
xor_data = xor_df.ix[:, 0:1] # 데이터
xor_label = xor_df.ix[:, 2] # 레이블
# print(xor_data)
# print(xor_label)

# 데이터 학습과 예측 하기
clf = svm.SVC()
clf.fit(xor_data, xor_label)
pre = clf.predict(xor_data)

# 정답률 구하기
ac_score = metrics.accuracy_score(xor_label, pre)
print('정답률', ac_score)
