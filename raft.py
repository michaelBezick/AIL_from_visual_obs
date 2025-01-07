from torchvision.models.optical_flow import raft_small
import time
from torchvision.utils import flow_to_image
import torch
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F

def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.savefig("predicted_flows.jpg")

def minmax(x: torch.Tensor):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))

device = "cuda"

model = raft_small(pretrained=True, progress=False).to(device)
model = model.eval()


img1 = cv2.imread("./Inputs/basketball1.png")
img2 = cv2.imread("./Inputs/basketball2.png")

time1 = time.time()

img1 = torch.from_numpy(img1).float().unsqueeze(0).to(device)
img2 = torch.from_numpy(img2).float().unsqueeze(0).to(device)

#need float beween [-1, 1]

img1 = minmax(img1) * 2 - 1
img2 = minmax(img2) * 2 - 1

img1 = torch.permute(img1, (0, 3, 1, 2))
img2 = torch.permute(img2, (0, 3, 1, 2))


transform = T.Compose(
    [
        T.Resize(size=(160, 160))
    ]
)

# img1 = transform(img1)
# img2 = transform(img2)

flow = model(img1, img2)

flow = flow[-1]

time2 = time.time()

print(flow.size())

flow = flow_to_image(flow)

img_batch = torch.concat([img1, img2], dim=0)

img = [(img + 1) / 2 for img in img_batch]

grid = [[img, flow_img] for (img, flow_img) in zip(img, flow)]

plot(grid)

print(time2 - time1)
