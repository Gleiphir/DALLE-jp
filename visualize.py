import matplotlib.pyplot as plt

loss = []

with open("output/vaeloss.csv") as F:
    for ln in F.readlines():
        loss.append( float(ln) )


plt.title("Sorted")
plt.ylabel("loss value")
plt.xlabel("number of samples")
plt.plot(sorted(loss[39000:41000],reverse=True))#
plt.vlines(x=(41000-39000)*0.025,ymin=0,ymax=0.2,label="2.5%",color="red")
plt.show()