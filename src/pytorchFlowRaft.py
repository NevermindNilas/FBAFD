import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights

# TO:DO TWEAK THAT PERCENTAGE STUFF
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained RAFT model once
weights = Raft_Large_Weights.DEFAULT
raftModel = raft_large(weights=weights).to(device).eval()


def prepareTensor(frame, precision):
    tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    if precision == "fp16":
        tensor = tensor.half()
    else:
        tensor = tensor.float()
    tensor = F.interpolate(
        tensor, size=(520, 960), mode="bilinear", align_corners=False
    )
    return tensor


@torch.inference_mode()
def pytorchFlowRaft(
    frame1, frame2, threshold=1.0, percentage=0.99999, precision: str = "fp32"
):
    tensor1 = prepareTensor(frame1, precision)
    tensor2 = prepareTensor(frame2, precision)

    if precision == "fp16":
        raftModel.half()
    else:
        raftModel.float()

    flow = raftModel(tensor1, tensor2)[-1]
    magnitude = torch.sqrt(flow[:, 0, :, :] ** 2 + flow[:, 1, :, :] ** 2)

    # Apply threshold to the magnitude of the flow vectors
    zeroFlowCount = torch.sum(magnitude < threshold).item()
    totalVectors = magnitude.numel()
    zeroFlowPercentage = zeroFlowCount / totalVectors

    # Return a BOOL value and the prediction value
    return zeroFlowPercentage > percentage, zeroFlowPercentage
