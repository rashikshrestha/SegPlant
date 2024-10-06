import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Loss based on DICE score to train UNet model.
    """
    def __init__(self, get_dice_iou=False):
        super().__init__()
        self.get_dice_iou = get_dice_iou

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        inputs_sum = inputs.sum()
        targets_sum = targets.sum()
        smooth_dice = (2.*intersection + smooth)/(inputs_sum + targets_sum + smooth)
        dice_loss = 1-smooth_dice

        # Get IoU and DICE scores
        if self.get_dice_iou:
            dice = (2.*intersection)/(inputs_sum + targets_sum)
            iou = intersection/(inputs_sum + targets_sum - intersection)
            return dice_loss, dice, iou

        return dice_loss