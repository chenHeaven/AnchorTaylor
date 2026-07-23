import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import kornia for SSIM, otherwise use a simple implementation or placeholder
try:
    import kornia
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False

class FusionWithDetectionLoss(nn.Module):
    """
    Implementation of the loss function described in:
    'Detection-guided Image Fusion'
    Includes:
    1. Fusion Loss (L_f) = L_grad + L_ther + L_stru
    2. Detection Loss (L_det) - passed from external YOLO loss
    3. Structural alignment Loss (Optional helper)
    """
    def __init__(self, eta1=20.0, eta2=10.0, eta3=5.0, lambda_fusion=0.5, lambda_det=1.0, device='cuda'):
        super().__init__()
        # Weights for Fusion Loss components (from paper or tuned)
        # Paper: eta1=0.5, eta2=1, eta3=1. 
        # Tuned: gradients are small, usually need larger weights to balance with detection loss
        self.eta1 = eta1 # Gradient
        self.eta2 = eta2 # Thermal
        self.eta3 = eta3 # Structure
        
        # Weights for Total Loss
        self.lambda_fusion = lambda_fusion
        self.lambda_det = lambda_det
        
        self.device = device
        
        if not HAS_KORNIA:
            print("Warning: Kornia not installed. SSIM calculation will be simplified.")

    def gradient_loss(self, img_c, img_f):
        """
        L_grad = 1 - GSI(d_c, d_f) - simplified to L1 difference of gradients
        Comparing gradient of fused image with gradient of input (typically Vis or max(Vis, IR))
        """
        def gradient(x):
            h_x = x.size(2)
            w_x = x.size(3)
            # x gradients
            grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
            # y gradients
            grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
            return grad_x, grad_y

        grad_c_x, grad_c_y = gradient(img_c)
        grad_f_x, grad_f_y = gradient(img_f)
        
        # We want the gradients to be similar
        loss_x = F.l1_loss(grad_f_x, grad_c_x)
        loss_y = F.l1_loss(grad_f_y, grad_c_y)
        
        return loss_x + loss_y

    def thermal_loss(self, img_ir, img_f):
        """
        L_ther = (TTI - 1)^2 - simplified to L1/MSE pixel intensity loss.
        Fused image should preserve thermal intensity from IR image.
        """
        # Simple pixel-wise loss to maintain thermal intensity
        return F.l1_loss(img_f, img_ir)

    def structure_loss(self, img_c, img_f):
        """
        L_stru = 1 - SSIM(I_c, I_f).
        """
        if HAS_KORNIA:
            # window_size should be odd
            ssim_val = kornia.metrics.ssim_loss(img_f, img_c, window_size=11, reduction='mean')
            return ssim_val # ssim_loss returns 1 - ssim
        else:
            # Fallback simple MSE structure loss if no kornia
            return F.mse_loss(img_f, img_c)

    def forward(self, i_vis, i_ir, i_f, l_det=0.0):
        """
        training step loss calculation.
        Args:
            i_vis: Visible input image/feature (B, C, H, W)
            i_ir: Infrared input image/feature (B, C, H, W)
            i_f: Fused output image/feature (B, C, H, W)
            l_det: Detection loss computed by YOLO head (scalar tensor)
        """
        # 1. Input for Gradient/Structure comparison
        # Paper suggests I_c. Often max(Vis, IR) or just Vis for structure.
        # Let's take element-wise max as the "ideal" structure reference
        i_c = torch.max(i_vis, i_ir)
        
        # 2. Compute components of Fusion Loss
        l_grad = self.gradient_loss(i_vis, i_f) # Gradient mostly from Vis
        l_ther = self.thermal_loss(i_ir, i_f)   # Thermal mostly from IR
        l_stru = self.structure_loss(i_c, i_f)  # Structure from combined
        
        l_fusion = self.eta1 * l_grad + self.eta2 * l_ther + self.eta3 * l_stru
        
        # 3. Total Loss
        # Detection guidance: L_det is optimized alongside fusion
        l_total = self.lambda_fusion * l_fusion + self.lambda_det * l_det
        
        return l_total, {
            "loss_fusion": l_fusion.detach(),
            "loss_grad": l_grad.detach(),
            "loss_ther": l_ther.detach(),
            "loss_stru": l_stru.detach(),
            "loss_det": l_det if isinstance(l_det, float) else l_det.detach()
        }
