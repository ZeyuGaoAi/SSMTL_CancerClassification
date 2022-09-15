import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from tqdm import tqdm
import numpy as np

from torchvision import models

from models import resynet

from load_patches_data import PatchesDataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

test_files = "./data/RCC/all_2000_test.txt"
file_path_base = "./path_to_model/model_best.pth.tar"
results_file = "./data/results.csv"
num_classes_1 = 2
num_classes_2 = 3

print(test_files)
print(file_path_base)

normMean = [0.744, 0.544, 0.670]
normStd = [0.183, 0.245, 0.190]
normTransform = transforms.Normalize(normMean, normStd)
test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        normTransform
    ])
	
test_data = PatchesDataset(test_files, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=False)

base_model = resynet.resnet34(pretrained=True, num_classes_1=num_classes_1, num_classes_2=num_classes_2)
base_model = base_model.cuda()
base_model = torch.nn.DataParallel(base_model)

checkpoint = torch.load(file_path_base)
base_model.load_state_dict(checkpoint['state_dict'])
#base_model.load_state_dict(checkpoint['net'])
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss().cuda()
base_model.eval()
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        f = open(results_file, 'ab')
        inputs, targets = inputs.to('cuda'), targets.type(torch.LongTensor).to('cuda')
        outputs_1, outputs_2 = base_model(inputs)
        np.savetxt(f, np.concatenate((outputs_1.cpu().data.numpy(),outputs_2.cpu().data.numpy()), axis=1), delimiter=",", fmt='%.4f')
        f.close()
