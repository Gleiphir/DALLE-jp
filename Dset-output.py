import torchvision.datasets as dset
import torchvision.transforms as transforms
from JPtokenize import token_dataset,fixlen
import torch
from dalle_pytorch import DiscreteVAE, DALLE
import numpy as np
from torch.utils.data.dataloader import DataLoader
import random
from torchvision.utils import save_image
from PIL import Image


IMAGE_SIZE = 256 # 256*256

NUM_TOKENS = 65536 # Larger than actually

TEXTSEQLEN = 80

BATCH_SIZE = 1

TRAIN_BATCHES = 100

#https://github.com/lucidrains/DALLE-pytorch/issues/33
#Edit: And yup, you need to reserve 0 for padding and 1 for , so add 2 to your encoded text ids!


cap = dset.CocoCaptions(root = './coco/images',
                        annFile = './coco/annotations/captions_val2014.json',
                        transform=transforms.Compose([
                            transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                            transforms.ToTensor(),
                            transforms.Normalize( (0.5,0.5,0.5), (0.5,0.5,0.5))
                        ]))
for i in range(100):
    img,_ = cap[i]


    #Dimg  = transforms.ToPILImage()(img)

    #print(images.size(),torch.min(Dimg),torch.max(Dimg),torch.mean(Dimg))
    save_image( img ,"./raw/{}.png".format(i))
    #Dimg.save("./raw/{}.png".format(i))



