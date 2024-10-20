import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def pytorchFlowRaft(
    frame1, frame2, threshold=1.0, percentage=0.95, precision: str = "fp32"
):
    tensor1 = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    tensor2 = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    # Load the pre-trained RAFT model
    weights = Raft_Large_Weights.DEFAULT
    raft_model = raft_large(weights=weights).to(device).eval()

    if precision == "fp16":
        raft_model.half()
        tensor1 = tensor1.half()
        tensor2 = tensor2.half()
    else:
        raft_model.float()
        tensor1 = tensor1.float()
        tensor2 = tensor2.float()

    # Requires frames of size 520x960, so resize the frames
    tensor1 = F.interpolate(
        tensor1, size=(520, 960), mode="bilinear", align_corners=False
    )
    tensor2 = F.interpolate(
        tensor2, size=(520, 960), mode="bilinear", align_corners=False
    )

    flow = raft_model(tensor1, tensor2)[-1]
    magnitude = torch.sqrt(flow[:, 0, :, :] ** 2 + flow[:, 1, :, :] ** 2)

    # Apply threshold to the magnitude of the flow vectors
    zeroFlowCount = torch.sum(magnitude < threshold).item()
    totalVectors = magnitude.numel()
    zeroFlowPercentage = zeroFlowCount / totalVectors

    # Return a BOOL value and the prediction value
    return zeroFlowPercentage > percentage, zeroFlowPercentage
