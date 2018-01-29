# mnist 데이터 csv 형태로 변환
import struct

def to_csv(name, maxdata):
    # 레이블 파일과 이미지 파일 열기
    lbl_f = open('./mnist/'+name+'-labels-idx1-ubyte', 'rb')
    img_f = open('./mnist/'+name+'-images-idx3-ubyte', 'rb')
    csv_f = open('./mnist/'+name+'.csv', 'w', encoding='utf-8')
    # 헤더 정보 읽기
    # 리틀 인디안 형태로 들어있음
    mag, lbl_count = struct.unpack('>II', lbl_f.read(8))
    mag, lmg_count = struct.unpack('>II', img_f.read(8))
    rows, cols = struct.unpack('>II', img_f.read(8))
    pixels = rows * cols

    # 이미지 데이터 읽고 csv로 저장하기
    res = []
    for idx in range(lbl_count):
        if idx > maxdata: break
        label = struct.unpack('B', lbl_f.read(1))[0]
        bdata = img_f.read(pixels)
        sdata = list(map(lambda n: str(n), bdata))
        csv_f.write(str(label)+",")
        csv_f.write(",".join(sdata) + '\r\n')
        # 이미지 파일로 테스트
        if idx < 10:
            s = 'P2 28  28  255\n'
            s += " ".join(sdata)
            iname = './mnist/{0}-{1}-{2}.pgm'.format(name, idx, label)

            with open(iname, 'w', encoding='utf-8') as f:
                f.write(s)
    csv_f.close()
    lbl_f.close()
    img_f.close()
# 결과를 파일로 출력
to_csv('train', 99999)
to_csv('t10k', 500)
