import torchvision.datasets as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root = './coco/images',
                        annFile = './coco/annotations/captions_val2014.json',)
                        #transform=transforms.ToTensor())

print('Number of samples: ', len(cap))
#img, target = cap[3] # load 4th sample
L = []

for i, (img, target) in enumerate(cap):
    for s in target:
        L.append("{}|{}\n".format(i,s))
        print("{}|{}".format(i,s))

with open("ann_export.txt","w",encoding='utf-8') as Fp:
    Fp.writelines(L)


