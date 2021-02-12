import torchvision.datasets as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root = 'dir where images are',
                        annFile = 'json annotation file',
                        transform=transforms.ToTensor())

print('Number of samples: ', len(cap))
#img, target = cap[3] # load 4th sample
L = []

for i, (img, target) in enumerate(cap):
    for s in target:
        L.append("{}|{}".format(i,s))
        
with open("ann_export.txt","w",encoding='utf-8') as Fp:
    Fp.writelines(L)


