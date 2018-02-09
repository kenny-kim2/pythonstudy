from InceptionModule import InceptionModule
import urllib.request

i = InceptionModule()
# i.init_data()

# i.start()


# imagedata = {}
# predictdata = []
# realprodlist = []
#
# try:
#     with open("./checklist_15253.txt", 'r') as f:
#         while True:
#             line = f.readline()
#             if not line: break
#             line = line.replace('\n', '')
#             line = line.replace(' ', '')
#             key, value, cmpny_code, prod_code, system_info = line.split('\t')
#
#             # try:
#             #     urllib.request.urlopen(value)
#             # except Exception:
#             #     print('testlist invalid url', value)
#             #     continue
#
#             imagedata[value] = key
#             predictdata.append(value)
#             realprodlist.append(key)
#         if len(imagedata) % 100 == 0:
#             print('check process in', str(len(imagedata)))
# except FileNotFoundError:
#     print('load error')
#
# print('predict data size =', len(predictdata))
# result = i.predict(predictdata)
# print('data return =', result)
#
# notsame = 0.
# for i in range(len(result)):
#     realprod = realprodlist[i]
#     predictprod = result[i]
#
#     if realprod != predictprod:
#         print('not same', predictdata[i], 'predict:', str(predictprod), 'real:', str(realprod))
#         notsame += 1
#
# print('오차율:', str(notsame/len(result) * 100), '%')


# imagedata = ['http://www.poom.co.kr/Upload2/Product/201609/1609070127_detail1.jpg']
# result = i.predict(imagedata)