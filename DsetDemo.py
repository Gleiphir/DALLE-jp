import torchvision.datasets as dset
import torchvision.transforms as transforms
from JPtokenize import token_dataset
import torch
from dalle_pytorch import DiscreteVAE, DALLE

IMAGE_SIZE = 256


cap = dset.CocoCaptions(root = './coco/images',
                        annFile = './coco/annotations/captions_val2014.json',
                        transform=transforms.Compose([
                            transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                            transforms.ToTensor(),
                        ]))

tokenDset = token_dataset('./coco/merged.txt')


print('Number of samples: ', len(cap))
#img, target = cap[3] # load 4th sample
L = []

print('Max len',tokenDset.maxLen())


for i, (img, target) in enumerate(cap):
    print(i,":",tokenDset.getRand(i),img.size())
    if i > 10:
        break

