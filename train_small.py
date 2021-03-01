import torchvision.datasets as dset
import torchvision.transforms as transforms
from JPtokenize import token_dataset,fixlen
import torch
from dalle_pytorch import DiscreteVAE, DALLE
import numpy as np
from torch.utils.data.dataloader import DataLoader
import random
from PIL import Image



IMAGE_SIZE = 256 # 256*256

NUM_TOKENS = 65536 # Larger than actually

TEXTSEQLEN = 80

BATCH_SIZE = 1

TRAIN_BATCHES = 100

#https://github.com/lucidrains/DALLE-pytorch/issues/33
#Edit: And yup, you need to reserve 0 for padding and 1 for , so add 2 to your encoded text ids!

DATASET_SIZE = 1000

EPOCHS = 100

learning_rate = 0.0002

vae = DiscreteVAE(
    image_size = IMAGE_SIZE,
    num_layers = 3,          # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
    num_tokens = 8192,       # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
    codebook_dim = 512,      # codebook dimension
    hidden_dim = 64,         # hidden dimension
    num_resnet_blocks = 1,   # number of resnet blocks
    temperature = 0.9,       # gumbel softmax temperature, the lower this is, the harder the discretization
    straight_through = False # straight-through for gumbel softmax. unclear if it is better one way or the other
).cuda()


optimizerVAE = torch.optim.Adam(vae.parameters(), lr=learning_rate)



"""
text = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, TEXTSEQLEN))
images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
mask = torch.ones_like(text).bool()
"""


cap = dset.CocoCaptions(root = './coco/images',
                        annFile = './coco/annotations/captions_val2014.json',
                        transform=transforms.Compose([
                            transforms.RandomCrop((IMAGE_SIZE,IMAGE_SIZE)),
                            #transforms.Resize((IMAGE_SIZE,IMAGE_SIZE),Image.NEAREST),
                            transforms.ToTensor(),
                            transforms.Normalize( (0.5,0.5,0.5), (0.5,0.5,0.5))
                        ]))


tokenDset = token_dataset('./coco/merged-1000.txt')

VAEloss = []

for epoch in range(EPOCHS):
    for i in range(DATASET_SIZE):
        #print(i,":",tokenDset.getRand(i),img.size())
        optimizerVAE.zero_grad()
        img,_ = cap[i]
        img=img.unsqueeze(0).cuda()
        #print(img.size())
        if i %10 == 0:
            print("VAE epoch {} / {}".format(i+ epoch* DATASET_SIZE,EPOCHS * DATASET_SIZE))
        loss = vae(img,return_recon_loss = True)
        VAEloss.append( loss.cpu().detach().numpy()  )
        loss.backward()
        optimizerVAE.step()

np.savetxt("vaeloss.csv",np.asarray(VAEloss),delimiter=",")

torch.save(vae.state_dict(),"Vae-small.pth")

dalle = DALLE(
    dim = 1024,
    vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
    num_text_tokens = NUM_TOKENS,    # vocab size for text
    text_seq_len = TEXTSEQLEN,         # text sequence length
    depth = 12,                 # should aim to be 64
    heads = 16,                 # attention heads
    dim_head = 64,              # attention head dimension
    attn_dropout = 0.1,         # attention dropout
    ff_dropout = 0.1            # feedforward dropout
).cuda()

optimizerDALLE= torch.optim.Adam(dalle.parameters(), lr=learning_rate)
DALLEloss = []

for epoch in range(EPOCHS):
    for i in range(DATASET_SIZE):
        #print(i,":",tokenDset.getRand(i),img.size())
        optimizerDALLE.zero_grad()
        img,strs = cap[i]
        #print(img.size())
        img = img.unsqueeze(0).cuda()
        if i % 10 == 0:
            print("DALLE epoch {} / {}".format(i+epoch*DATASET_SIZE, EPOCHS * DATASET_SIZE))
        try:
            textToken, mask = fixlen([tokenDset.getRand(i)])
        except KeyError:
            continue
        loss = dalle(textToken.cuda(), img, mask=mask.cuda(), return_loss=True)
        DALLEloss.append(loss.detach().cpu().numpy())
        loss.backward()
        optimizerDALLE.step()

np.savetxt("dalleloss.csv",np.asarray(DALLEloss),delimiter=",")



# do the above for a long time with a lot of data ... then

torch.save(dalle.state_dict(),"dalle-small.pth")

test_text = "犬が地面に寝そべっている写真"

textToken, mask = fixlen( [tokenDset.tokenizeList(test_text) ])

images = dalle.generate_images(textToken.cuda(), mask = mask)
print(images.shape) # (2, 3, 256, 256)


