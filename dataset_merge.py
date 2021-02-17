import regex as re

Jp = re.compile(r"^[0-9]+\|.*([\p{Han}|\p{Hiragana}|\p{Katakana}]).*$",re.UNICODE )

fileL = [r"C:\Users\zhang\Downloads\split 15 parts\{} JA.txt".format(i) for i in range(15)]

with open(r"C:\Users\zhang\Downloads\split 15 parts\merged.txt",mode='w',encoding='utf-8') as Fpw:
    L = []
    for f in fileL:
        with open(f,encoding='utf-8') as Fp:
            for line in Fp.readlines():
                if len(line) <=0:
                    continue
                line = line.replace("｜","|")
                if Jp.search(line):
                    L.append(line)
                else:
                    print(line)
        print(f)
    Fpw.writelines(L)
print("done")
#res = Jp.search(s)
#print(res)

