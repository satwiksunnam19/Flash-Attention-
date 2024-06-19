import torch 
import torch.nn as nn 

from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

class CrossEntropyLoss(nn.Module):
    def __init__(
            self,
            ignore_index=False, 
            reduction="mean",
            label_smoothing=0.0, 
            logit_scale=1.0, 
            lse_square_scale=0.0,
            inplace_backward=False,
            process_group=None,
            return_z_loss=False
    ):
        """
        Args: 
        ignored_index: True If labels==ignored_index, the loss is set to 0.0
        label_smoothing: float 
        lse_square_scale: float. if > 0, we add lse_square_scale*lse(logits)^2 to the loss. This is also called as "z-loss"
        inplace_backward: bool. If True, we do the backward pass inplace by modifying the logits this saves the memory, 
        process_group: if not None, we're doing Tensor parallel: each process is responsible for one part of the vocab. The loss is aggregated by the processes.
        return_z_loss: bool. If True, we retrun the component of the loss contributed by the lse_square_scale value. This value is only for logging and does not suppourt backprop.
        """

        super().__init__()
        if reduction not in ["mean","none","sum"]:
            raise NotImplementedError("only suppourt reduction= 'mean' or 'none' or 'sum' ")
        
        self.ignore_index=ignore_index
        self.label_smoothing=label_smoothing
        self.logit_scale=logit_scale
        self.lse_square_scale=lse_square_scale
        self.inplace_backward=inplace_backward
        self.process_group=process_group
        self.return_z_loss=return_z_loss


    def forward(self,input,target):
        """
        Arguments: 
        input : (batch,vocab_size)
        target: (batch)

        returns : 
        losses: (batch,) if reduction is 'None' else (1,), dtype float.
        z_loss: (batch,) if reduction is 'None' else (1,), dtype float (if self.return_z_loss) 
        """

        assert input.is_cuda and target.is_cuda , "Only suppourts CUDA tensors"

        loss,z_loss=cross_entropy_loss(
            input,target,label_smoothing=self.label_smoothing,
            logit_scale=self.logit_scale, lse_square_scale=self.lse_square_scale,
            ignore_index=self.ignore_index, inplace_backward=self.inplace_backward, process_group=self.process_group
    
        )

        if self.reduction=="mean":
            loss=loss.sum()/(target !=self.ignore_index).sum()
        
        if self.reduction=="sum": 
            loss=loss.sum()
        
        else:
            loss=loss
        
        if not self.return_z_loss: 
            return loss 
        
        if self.reduction=="mean": 
            z_loss=z_loss.sum()/(target !=self.ignore_index).sum()
        
        elif self.reduction=="sum":
            z_loss=z_loss.sum()
        
        else: 
            z_loss=z_loss
        
        return loss,z_loss
        
