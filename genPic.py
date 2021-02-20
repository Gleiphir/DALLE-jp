import torchvision.datasets as dset
import torchvision.transforms as transforms
from JPtokenize import token_dataset,fixlen
import torch
from dalle_pytorch import DiscreteVAE, DALLE
import numpy as np
from torch.utils.data.dataloader import DataLoader
import random
from torchvision.utils import save_image

IMAGE_SIZE = 256 # 256*256

NUM_TOKENS = 65536 # Larger than actually

TEXTSEQLEN = 80

BATCH_SIZE = 1

TRAIN_BATCHES = 100

#https://github.com/lucidrains/DALLE-pytorch/issues/33
#Edit: And yup, you need to reserve 0 for padding and 1 for , so add 2 to your encoded text ids!






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

vae.load_state_dict(torch.load("Vae.pth"))

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

dalle.load_state_dict(torch.load("dalle.pth"))

"""
text = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, TEXTSEQLEN))
images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
mask = torch.ones_like(text).bool()
"""


tokenDset = token_dataset('./coco/merged.txt')


# do the above for a long time with a lot of data ... then

num_pics = 30

def denorm(img:torch.Tensor):
    mean = torch.mean(img)
    min_maxrange = torch.max(img) - torch.min(img)
    return ( (img - mean) / (min_maxrange / 2.0) + 0.5 )* 255

for i in range(30):

    test_text = "犬が地面に寝そべっている写真"

    textToken, mask = fixlen( [tokenDset.tokenizeList(test_text) ])
    textToken = textToken.cuda()
    mask = mask.cuda()
    images = dalle.generate_images(textToken, mask = mask)
    Dimg  = denorm(images)
    print(images.size(),torch.min(Dimg),torch.max(Dimg),torch.mean(Dimg))
    save_image( Dimg ,"./imgs/{}.png".format(i),normalize=True)



