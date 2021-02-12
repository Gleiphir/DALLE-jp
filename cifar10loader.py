import torchvision
import torchvision.datasets as DSETS
from torch.utils.data.dataset import Dataset
import base64



labels = ["hikouki",
          "kuruma",
          "tori",
          "neko",
          "sika",
          "inu",
          "kaeru",
          "uma",
          "senpaku",
          "torakku",]



class Base64Cifar10(Dataset):
    def __init__(self):
        self.dset_cifar = DSETS.CIFAR10('./dataset')



    def __len__(self):
        return self.dset_cifar.__len__()

    def __getitem__(self, item):
        # random
        im,target = self.dset_cifar[item]
        return base64.b64encode(im.tobytes()).decode('utf-8'),labels[target]



if __name__ =='__main__':
    dset = Base64Cifar10()
    print(dset[0])
    print(len(dset[0][0]))