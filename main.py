import torchvision.datasets as dset
import torchvision.transforms as transforms
from JPtokenize import token_dataset


cap = dset.CocoCaptions(root = './coco/images',
                        annFile = './coco/annotations/captions_val2014.json',)
                        #transform=transforms.ToTensor())

tokenDset = token_dataset('./coco/merged.txt')


print('Number of samples: ', len(cap))
#img, target = cap[3] # load 4th sample
L = []

print('Max len',tokenDset.maxLen())

for i, (img, target) in enumerate(cap):
    print(i,":",tokenDset.getRand(i),img.size())
    if i > 10:
        break