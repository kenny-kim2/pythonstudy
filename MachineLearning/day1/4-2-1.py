from sklearn import svm

xor_data = [#P, Q, result
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

# 학습을 위해 데이터와 레이블 분리
data = []
label = []
for row in xor_data:
    p = row[0]
    q = row[1]
    r = row[2]
    data.append([p, q])
    label.append(r)

# 데이터 학습 시키기
clf = svm.SVC()
clf.fit(data, label)

# 데이터 예측 하기
pre = clf.predict(data)
print('예측 결과:', pre)

# 결과 확인하기
ok = 0
total = 0

for idx, answer in enumerate(label):
    p = pre[idx]
    if p == answer:
        ok += 1
    total += 1

print('정답률:', ok, '/', total, '=', ok/total)