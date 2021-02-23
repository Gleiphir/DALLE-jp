
NUM_FIG = 1000

lines = []

with open("merged.txt",encoding='utf-8') as F:
    for ln in F.readlines():
        try:
            ind_t, text_t = ln.split("|")
            ind = int(ind_t)
            if ind > NUM_FIG:
                break
            lines.append( str(ind) + "|" + text_t  )
        except Exception as e:
            print(e)
            continue

with open("merged-{}.txt".format(NUM_FIG),"w",encoding='utf-8') as oF:
    oF.writelines(lines)