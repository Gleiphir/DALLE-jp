import torchvision.datasets as dset
import torchvision.transforms as transforms
from JPtokenize import token_dataset
import torch
from dalle_pytorch import DiscreteVAE, DALLE
import numpy as np
from torch.utils.data.dataloader import DataLoader
import random


IMAGE_SIZE = 256 # 256*256

NUM_TOKENS = 65536 # Larger than actually

TEXTSEQLEN = 80

BATCH_SIZE = 1

TRAIN_BATCHES = 100

#https://github.com/lucidrains/DALLE-pytorch/issues/33
#Edit: And yup, you need to reserve 0 for padding and 1 for , so add 2 to your encoded text ids!




def fixlen(orig:list):
    fix = np.vectorize(lambda n: n + 2 if n != NUM_TOKENS else 0)
    lens = [len(l) for l in orig]
    data = np.full( (len(orig),TEXTSEQLEN),NUM_TOKENS )
    Mask = np.arange(TEXTSEQLEN) < np.array(lens)[:, None]
    data[Mask] = np.concatenate(orig)
    return torch.Tensor(fix(data)),torch.Tensor(Mask)



vae = DiscreteVAE(
    image_size = IMAGE_SIZE,
    num_layers = 3,          # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
    num_tokens = 8192,       # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
    codebook_dim = 512,      # codebook dimension
    hidden_dim = 64,         # hidden dimension
    num_resnet_blocks = 1,   # number of resnet blocks
    temperature = 0.9,       # gumbel softmax temperature, the lower this is, the harder the discretization
    straight_through = False # straight-through for gumbel softmax. unclear if it is better one way or the other
)





"""
text = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, TEXTSEQLEN))
images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
mask = torch.ones_like(text).bool()
"""


cap = dset.CocoCaptions(root = './coco/images',
                        annFile = './coco/annotations/captions_val2014.json',
                        transform=transforms.Compose([
                            transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                            transforms.ToTensor(),
                        ]))

loader = DataLoader(cap)

tokenDset = token_dataset('./coco/merged.txt')


for i, (img, target) in enumerate(loader):
    #print(i,":",tokenDset.getRand(i),img.size())
    print("VAE epoch {} / {}",i,len(loader))
    loss = vae(img,return_recon_loss = True)
    loss.backward()

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
)


loader = DataLoader(cap)
for i, (img, target) in enumerate(loader):
    print("DALLE epoch {} / {}", i, len(loader))
    im = fixlen(tokenDset.getRand(i))
    textToken, mask = fixlen( tokenDset.tokenizeList(random.choice(target)) )
    loss = dalle(textToken, im, mask = mask, return_loss = True)
    loss.backward()

# do the above for a long time with a lot of data ... then

test_text = "犬が地面に寝そべっている写真"

textToken, mask = fixlen( tokenDset.tokenizeList(test_text) )

images = dalle.generate_images(textToken, mask = mask)
print(images.shape) # (2, 3, 256, 256)


