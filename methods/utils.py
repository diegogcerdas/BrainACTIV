import torch

def resize(measure, size):
    measure = torch.from_numpy(measure).float().unsqueeze(0)
    measure = torch.nn.functional.interpolate(
        measure,
        size=(size,size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    measure = measure.numpy()
    return measure