import torch

################################################################################

def mse_penalty(x, gam: float = None, reduce="mean"):
    """
      ⡤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤
    4 ⡇⠳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠞⢸
      ⡇⠀⠈⢧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡼⠁⠀⢸
      ⡇⠀⠀⠀⠙⢦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⣀⡴⠋⠀⠀⠀⢸
      ⡇⠀⠀⠀⠀⠀⠈⠳⢄⡀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⢀⡠⠞⠁⠀⠀⠀⠀⠀⢸
    0 ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠉⠓⠦⢤⣀⣀⣀⣇⣀⣀⡤⠴⠚⠉⠀⠀⠀⠀⠀⠀⠀⠀⢸
      ⠓⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠚
      ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀
     Args:
         x:   array (last dimenion is reduced, all other dimenions are averaged)
         gam: the scaling of the penalty
         reduce: ["mean", "sum" or nothing] whether to reduce the penalties
     Returns:
         the penalty, possibly reduced
    """
    gam = 1e2 if gam is None else gam
    ret = gam * torch.sum(x ** 2, dim=-1)
    if reduce.lower() == "mean":
        return torch.mean(ret)
    elif reduce.lower() == "sum":
        return torch.sum(ret)
    else:
        return ret


def exact_penalty(x, gam: float = None, reduce="mean"):
    """
      ⡤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤
    2 ⡇⠙⠢⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠔⠋⢸
      ⡇⠀⠀⠀⠙⠢⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⣠⠔⠋⠀⠀⠀⢸
      ⡇⠀⠀⠀⠀⠀⠀⠙⠢⣄⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⣠⠔⠋⠀⠀⠀⠀⠀⠀⢸
      ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠢⣄⠀⠀⠀⡇⠀⠀⣠⠔⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸
    0 ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠢⣄⣧⠔⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸
      ⠓⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠚
      ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀
     Args:
         x:   array (last dimenion is reduced, all other dimenions are averaged)
         gam: the scaling of the penalty
         reduce: ["mean", "sum" or nothing] whether to reduce the penalties
     Returns:
         the penalty, possibly reduced
    """
    gam = 1e1 if gam is None else gam
    ret = gam * torch.norm(x, dim=-1)
    if reduce.lower() == "mean":
        return torch.mean(ret)
    elif reduce.lower() == "sum":
        return torch.sum(ret)
    else:
        return ret


def super_exact_penalty(x, gam: float = None, reduce="mean"):
    """
      ⡤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤
    2 ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸
      ⡇⠦⠤⣄⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣠⠤⠴⢸
      ⡇⠀⠀⠀⠀⠉⠙⠒⠢⢤⣀⡀⠀⠀⠀⠀⡇⠀⠀⠀⢀⣀⡤⠔⠒⠋⠉⠀⠀⠀⠀⢸
      ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠓⠦⣄⠀⡇⣠⠴⠚⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸
    0 ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢳⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸
      ⠓⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠚
      ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2
     Args:
         x:   array (last dimenion is reduced, all other dimenions are averaged)
         gam: the scaling of the penalty
         reduce: ["mean", "sum" or nothing] whether to reduce the penalties
     Returns:
         the penalty, possibly reduced
    """
    gam = 1e0 if gam is None else gam
    ret = gam * torch.sqrt(torch.norm(x, dim=-1))
    if reduce.lower() == "mean":
        return torch.mean(ret)
    elif reduce.lower() == "sum":
        return torch.sum(ret)
    else:
        return ret

################################################################################
