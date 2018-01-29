from sklearn import svm, metrics
import glob, os.path, re, json

# 텍스트를 읽어 출현빈도 조사
def check_freq(fname):

    name = os.path.basename(fname)
    print(name)
    lang = re.match(r'^[a-z]{2,}', name).group()

check_freq('./lang/test/en-1.txt')