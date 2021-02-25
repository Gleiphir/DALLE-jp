import matplotlib.pyplot as plt

loss = []

with open("output/dalleloss.csv") as F:
    for ln in F.readlines():
        loss.append( float(ln) )

plt.plot(loss)
plt.show()