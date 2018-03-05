from konlpy.tag import Twitter

twitter = Twitter()

data = '[KB국민카드7%할인,26일 단하루]1300k 요넥스로 나노레이 NR20 배드민턴라켓 블루'

print(twitter.pos(data, norm=True))

result = []
for data in twitter.pos(data, norm=True):
    if data[1] == 'Foreign':
        print(data[1])
        continue
    if data[1] == 'Punctuation':
        print(data[1])
        continue
    if data[1] == 'Josa':
        print(data[1])
        continue
    result.append(data[0])

print(result)