# operate_model.py
import inspect
import torch
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import matplotlib.gridspec as gridspec
from captum.attr import IntegratedGradients
import math

def get_model_inputs(model, print_sig=True):
    sig = inspect.signature(model.forward)
    if print_sig:
        print(f"Model forward signature: {sig}")
    return sig

def automate_ig_analysis(model, batch, target_index=0, n_steps=50, image_dict=None):
    """
    Automatically calculates and visualizes the integrated gradients of the model for a given batchof inputs

    Parameters:
        model: A model that has been initialized and is set to eval mode
        batch: A batch obtained from the DataLoader. The input order
               must conform to the forward input order of the model,such as tuple/list.
               The order should be consistent with the parameters displayed by get_model_inputs(model) 
        target_index: The target output index for which attribution is to be calculated (eg. the index of a specific cell type)
        n_steps: The number of steps in the IG computation  (more steps result in more accurate computation, but also increase computational cost)
        image_dict: The dict specifies whether each parameter name is of type image.
                    For example, {'tile': True, 'subtiles': True, 'neighbors': True, 'coords': False}.If not provided, all are assumed to be non - images
                    
                    
    Return:
        A dictionary where the keys are the names of the input parameters and the values are the corresponding attribution tensors(while still retaining the batch dimension)
    """
    model.eval()
    
    # Parameter names of forward model 
    sig = get_model_inputs(model, print_sig=True)
    param_names = list(sig.parameters.keys())
    if 'self' in param_names:
        param_names.remove('self')
    print("Model input parameters:", param_names)
    
    # Retrieve inputs from the batch in sequence according to param_names
    if isinstance(batch, dict):
        inputs_list = [batch[name] for name in param_names]
    else:
        # Assuming it's a tuple/list，the order should match param_names 
        inputs_list = list(batch)
    
    # Use the first sample in the batch as an example for IG analysis,keeping the batch dimension=1
    inputs_sample = [inp[0:1] for inp in inputs_list]
    # Establish baseline：using all zero tensors
    baselines = [torch.zeros_like(inp) for inp in inputs_sample]

    # Using Captum's  IntegratedGradients-Calculating Attribution
    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        inputs=tuple(inputs_sample),
        baselines=tuple(baselines),
        target=target_index,
        n_steps=n_steps
    )
    
    # Compose the output dict，with parameter names corresponding to their attributes (preserving the batch dim)
    attr_dict = dict(zip(param_names, attributions))
    
    # Integrating the visualization: Create a separate block for each parameter
    num_params = len(param_names)
    fig = plt.figure(figsize=(10, num_params * 5))
    outer_gs = gridspec.GridSpec(num_params, 1, hspace=0.5)
    
    for i, name in enumerate(param_names):
        # Remove batch dim
        attr = attr_dict[name].squeeze(0)
        attr_np = attr.detach().cpu().numpy()
        
        ax = None  
        # Check if it's image data; default to False if image_dict is not
        is_image = image_dict.get(name, False) if image_dict is not None else False
        
        if is_image:
            # Process image input
            if attr_np.ndim == 3:
                # Shape (C, H, W) --> Single
                ax = fig.add_subplot(outer_gs[i])
                avg_attr = np.mean(attr_np, axis=0)
                im = ax.imshow(avg_attr, cmap='viridis')
                ax.set_title(f"Attribution for {name} (avg over channels)")
                fig.colorbar(im, ax=ax)
            elif attr_np.ndim == 4:
                # Shape (N, C, H, W) --> Multiple images，where N is the number of images
                N = attr_np.shape[0]
                grid_cols = math.ceil(math.sqrt(N))
                grid_rows = math.ceil(N / grid_cols)
                # Create an inner GridSpec at the outer position
                sub_gs = gridspec.GridSpecFromSubplotSpec(grid_rows, grid_cols, subplot_spec=outer_gs[i], 
                                                          wspace=0.3, hspace=0.3)
                # Iteratively display each picture
                for j in range(N):
                    ax_sub = fig.add_subplot(sub_gs[j])
                    avg_attr = np.mean(attr_np[j], axis=0)
                    im = ax_sub.imshow(avg_attr, cmap='viridis')
                    ax_sub.set_title(f"{name} Image {j}")
                    ax_sub.axis('off')
                    fig.colorbar(im, ax=ax_sub, fraction=0.046, pad=0.04)
            else:
                ax = fig.add_subplot(outer_gs[i])
                ax.text(0.5, 0.5, f"Unable to visualize shape: {attr_np.shape}", 
                        horizontalalignment='center', fontsize=12)
                ax.set_title(f"Attribution for {name}")
        else:
            # Non image data：presented as a bar chart (after flattening)
            ax = fig.add_subplot(outer_gs[i])
            flat_attr = attr_np.flatten()
            ax.bar(range(len(flat_attr)), flat_attr, color='skyblue')
            ax.set_title(f"Attribution for {name} (flattened)")
            
    plt.tight_layout()
    plt.show()
    
    return attr_dict

  
def make_input_to_device(model, batch, device, label_key="label", need_label=True):
    """
    Based on the parameter list of model.forward automatically extract input from the batch and move it to the specified device.If
    the parameter label_key is set,it will also try to extract data for that key
    If the label_key data cannot be found in the batch,a KeyError message will be thrown indicating that "your dataset doesn't contain a label
    or you need to change it to ur prediction.
    """
    # Obtain parameter signature of model.forward （exclusing self）
    sig = get_model_inputs(model, print_sig=False)
    # Store the parameters to be passed to forward 
    inputs = {}
    for name, _ in sig.parameters.items():
        if name == 'self':
            continue

        if name in batch:
            inputs[name] = batch[name].to(device)
        else:
            raise KeyError(f"Model requires {sig}. No data for '{name}'can be found in the batch.Please confirm that the output key of the dataloader matches the name of the model.forward parameter.")
    
    # Special handling for label_key ：If label_key is defined but not in inputs 
    if need_label:
        if label_key in batch:
            label = batch[label_key].to(device)
        else:
            raise KeyError(f"Your dataset doesn't contain '{label_key}' data or you need to change it yourself.")
    else:
        label = None
    return inputs, label



def predict(model, dataloader, device, **kwargs):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Using get_model_inputs to extract the input
            inputs, _ = make_input_to_device(model, batch, device, need_label=False)
            out = model(**inputs)
            all_preds.append(out.cpu())

    
    return torch.cat(all_preds).numpy()

class EarlyStopping:
    def __init__(self, patience=10, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_score is None or val_loss < self.best_score:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

class spear_EarlyStopping:
    def __init__(self, patience=10, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_score is None or val_loss > self.best_score:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def plot_losses(train_losses, val_losses, ax=None, title="Training vs Validation Loss"):
    """
    Plot train/validation loss curve
    :param train_losses: list or array,the numerical values of the training loss 
    :param val_losses: list or array,Validation of loss values
    :param ax: (optional) Pass in a matplotlib Axes object;if not provided,a plot will be created automatically
    :param title: (string) Chart title
    """
    # If ax is not present，create a new figure with ax,and use clear_output to avoid overlap
    if ax is None:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(train_losses, label="Train Loss", marker='o')
    ax.plot(val_losses, label="Val Loss", marker='o')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    ax.set_title(title)
# Collect information
def plot_per_cell_metrics(mse_vals, spearman_vals, cell_names=None, top_k=5, ax_mse=None, ax_spearman=None):
    """
    Plot the MSE and Spearman histogram for each cell type.
    Two axes can be specified for custom layout and plot_losses allignment.

    Params:
        mse_vals: array-like, MSE for each cell type 
        spearman_vals: array-like,Spearman values for each cell type 
        cell_names: list of str, cell type name(default is C1 ~ C35)
        top_k: int,the number of best/worst items to be worked
        ax_mse: matplotlib Axes,Use to plot MSEs
        ax_spearman: matplotlib Axes, used to plot Spearman diagrams.
    """
    if cell_names is None:
        cell_names = [f"C{i+1}" for i in range(len(mse_vals))]

    sorted_idx_mse = np.argsort(mse_vals)
    sorted_idx_spearman = np.argsort(spearman_vals)

    if ax_mse is None or ax_spearman is None:
        fig, (ax_mse, ax_spearman) = plt.subplots(1, 2, figsize=(14, 4))

    # Left: MSE per gene
    ax_mse.clear()
    ax_mse.bar(cell_names, mse_vals, color='skyblue')
    ax_mse.set_title("Per-cell MSE")
    ax_mse.tick_params(axis='x', rotation=45)
    for i in sorted_idx_mse[:top_k]:
        ax_mse.text(i, mse_vals[i] + 0.01, "↓", ha='center', color='red')
    for i in sorted_idx_mse[-top_k:]:
        ax_mse.text(i, mse_vals[i] + 0.01, "↑", ha='center', color='green')

    # Right: Spearman per gene
    ax_spearman.clear()
    ax_spearman.bar(cell_names, spearman_vals, color='orange')
    ax_spearman.set_title("Per-cell Spearman")
    ax_spearman.tick_params(axis='x', rotation=45)
    for i in sorted_idx_spearman[:top_k]:
        ax_spearman.text(i, spearman_vals[i] + 0.01, "↓", ha='center', color='red')
    for i in sorted_idx_spearman[-top_k:]:
        ax_spearman.text(i, spearman_vals[i] + 0.01, "↑", ha='center', color='green')

    if ax_mse is None or ax_spearman is None:
        plt.tight_layout()
        plt.show()

import numpy as np
import matplotlib.pyplot as plt



def get_alpha(epoch, initial_alpha=0.3, final_alpha=0.8, target_epoch=50, method="linear"):
    """
    The alpha value is calculated based on the current epoch and the specified method
    When epoch >= target_epoch , final_alpha is returned directly.

    :param epoch: Current epoch（Integer）
    :param initial_alpha: Initial alpha value
    :param final_alpha: The final alpha value(reached at target_epoch ）
    :param target_epoch: The epoch at which alpha is expected to reach final_alpha
    :param method: The scheduling method,options are "linear", "exponential", "cosine", "log"
    :return: The alpha value of the current epoch 
    """
    if not isinstance(epoch, (int, float)):
        raise TypeError(f"`epoch` must be int or float, but got {type(epoch)}")

    if epoch >= target_epoch:
        return final_alpha

    progress = epoch / target_epoch  # 進度比例 (0 ~ 1)
    if method == "linear":
        # Linear increase：when epoch==target_epoch, alpha = final_alpha
        return initial_alpha + (final_alpha - initial_alpha) * progress
    elif method == "exponential":
        # Exponential rise：Slower changes in the early stages，faster changes in the later stages
        return initial_alpha * ((final_alpha / initial_alpha) ** progress)
    elif method == "cosine":
        # Cosine attenuation：Smooth transition using a cosine curve
        # When epoch==0 then：cos(0)=1, alpha = final_alpha + (initial_alpha-final_alpha)*1 = initial_alpha
        # When epoch==target_epoch then：cos(pi)= -1, alpha = final_alpha + (initial_alpha-final_alpha)*0 = final_alpha
        return final_alpha + (initial_alpha - final_alpha) * (np.cos(np.pi * progress) + 1) / 2
    elif method == "log":
        # Logarithmic type：Use log₂ transition，and final_alpha will be reached exactly when epoch==target_epoch 
        return initial_alpha + (final_alpha - initial_alpha) * np.log2(1 + epoch) / np.log2(1 + target_epoch)
    else:
        raise ValueError(f"Unknown method: {method}")

import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_ranking_loss(pred: torch.Tensor,
                          target: torch.Tensor,
                          margin: float = 1.0) -> torch.Tensor:
    """
    Pred/target: (B, C)  B=batch_size, C=cell types
    For each sample i,enumerate all pairs j<k ：
      If target_ij > target_ik,then we expect pred_ij + margin > pred_ik
      Otherwise pred_ik + margin > pred_ij
    loss = mean( max(0, margin - sign(target_j - target_k) * (pred_j - pred_k)) )
    """
    B, C = pred.shape

    # 1) Construct the difference for all pairs (j,k).
    #    pred_diff: (B, C, C) where pred_diff[:, j, k] = pred[:, j] - pred[:, k]
    pred_diff = pred.unsqueeze(2) - pred.unsqueeze(1)      # B×C×C
    targ_diff = target.unsqueeze(2) - target.unsqueeze(1)  # B×C×C

    # 2)We only take the upper triangular part (j<k)to avoid repetition，and reflexivity
    idxs = torch.triu_indices(C, C, offset=1)
    pd = pred_diff[:, idxs[0], idxs[1]]   # shape (B, M) M=C*(C-1)/2
    td = targ_diff[:, idxs[0], idxs[1]]

    # 3) Calculate hinge loss：If the true td>0，we want pd>0，loss = max(0, margin - pd)
    #                          If the true td<0，we want pd<0，loss = max(0, margin + pd)
    sign = torch.sign(td)  # +1 or -1 or 0
    # margin - sign * pd
    raw = margin - sign * pd
    loss_pairs = torch.clamp(raw, min=0.0)

    # 4) Average
    return loss_pairs.mean()

def pearson_corr_loss(pred, target, eps=1e-8):
    # center
    p = pred - pred.mean(dim=1, keepdim=True)
    t = target - target.mean(dim=1, keepdim=True)
    # covariance / (σₚ·σₜ)
    num = (p * t).sum(dim=1)
    den = p.norm(dim=1) * t.norm(dim=1) + eps
    corr = num / den
    return 1.0 - corr.mean()

def pairwise_logistic_loss(pred: torch.Tensor,
                           target: torch.Tensor) -> torch.Tensor:
    """
    RankNet style pairwise logistic loss.
    pred/target: (B, C)
    For each sample i,enumerate all pairs j<k ：
      If target[i,j] > target[i,k],then label y_{jk}=1;otherwise y_{jk}=0.
    Calculate pd = pred_j - pred_k, then use BCEwithLogits(pd, y).
    """
    B, C = pred.shape
    # All j<k pairs of idx
    idxs = torch.triu_indices(C, C, offset=1)
    # pred_j - pred_k, target_j - target_k  -> (B, M)
    pd = (pred.unsqueeze(2) - pred.unsqueeze(1))[:, idxs[0], idxs[1]]
    td = (target.unsqueeze(2) - target.unsqueeze(1))[:, idxs[0], idxs[1]]
    # labels: 1 if td>0 else 0
    labels = (td > 0).float()
    # binary cross‐entropy with logits
    loss = F.binary_cross_entropy_with_logits(pd, labels, reduction='mean')
    return loss
from torchsort import soft_rank

def spearman_loss(pred, target):
    pred = pred.cpu()
    target = target.cpu()
    
    pred_ranks = soft_rank(pred, regularization_strength=1.0, regularization="l2")
    target_ranks = soft_rank(target, regularization_strength=1.0, regularization="l2")


    pred_centered = pred_ranks - pred_ranks.mean(dim=1, keepdim=True)
    target_centered = target_ranks - target_ranks.mean(dim=1, keepdim=True)

    covariance = (pred_centered * target_centered).sum(dim=1)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1))
    target_std = torch.sqrt((target_centered ** 2).sum(dim=1))
    corr = covariance / (pred_std * target_std + 1e-8)

    return 1 - corr.mean()

def hybrid_loss(pred: torch.Tensor,
                target: torch.Tensor,
                alpha: float = 0.5,
                loss_type: str = 'pearson',
                margin: float = 1.0) -> torch.Tensor:
    """
    A hybrid between MSE and a ranking-based loss.

    Args:
        pred, target: (B, C) Tensors.
        alpha: weight on the ranking loss (0 <= alpha <= 1).
        loss_type: 'pearson' or 'pairwise'.
        margin: hinge margin for pairwise ranking loss.
    Returns:
        (1-alpha)*MSE + alpha*RankingLoss
    """

    mse = F.mse_loss(pred, target)
    if loss_type == 'pearson':
        rank_loss = pearson_corr_loss(pred, target)
        loss = mse * (1 - alpha + alpha * rank_loss)
    elif loss_type == 'pairwise':
        rank_loss = pairwise_ranking_loss(pred, target, margin=margin)
        loss = (1 - alpha) * mse + alpha * rank_loss
        
    elif loss_type == 'logistic':
            rank_loss = pairwise_logistic_loss(pred, target)
            loss = (1 - alpha) * mse + alpha * rank_loss
    elif loss_type == 'weighted':
        weighting = 1.0 + target.abs()
        weighted_mse = weighting * mse
        loss = weighted_mse.mean()
    elif loss_type == 'soft_rank':
        rank_loss = spearman_loss(pred, target)
        loss = (1 - alpha) * mse * 10 + alpha * rank_loss
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type!r}")

    return loss



# =======================
# Improved version train_one_epoch and evaluate
# =======================

def train_one_epoch(model, dataloader, optimizer, device, current_epoch,
                    initial_alpha=0.3, final_alpha=0.9, target_epoch=15, method="linear", loss_type = 'pearson'):
    """
    Train for one epoch,and use dynamic alpha to calculate the hybrid loss。
    Only keep necessary parameters:
      - model, dataloader, optimizer, device, current_epoch, total_epochs
    Additionally,using the default settings: initial_alpha=0.3, final_alpha=0.8, when epoch >= target_epoch (default 50), alpha is fixed at final_alpha,
      Furthermore, beta is fixed at 1.0,and the scheduling method is determined by method
    
    :return: Average loss and Average Spearman correlation
    """
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    # Based on current epoch Calculate dynamic alpha
    alpha = get_alpha(current_epoch, initial_alpha, final_alpha, target_epoch, method)
    
    for batch in pbar:
        inputs, label = make_input_to_device(model, batch, device)
        optimizer.zero_grad()
        
        out = model(**inputs)
        loss = hybrid_loss(out, label, alpha=alpha, loss_type=loss_type)
        loss.backward()
        optimizer.step()
        
        batch_size = label.size(0)
        total_loss += loss.item() * batch_size
        
        all_preds.append(out.cpu())
        all_targets.append(label.cpu())
        
        avg_loss = total_loss / ((pbar.n + 1) * dataloader.batch_size)
        pbar.set_postfix(loss=loss.item(), avg=avg_loss)
    
    all_preds = torch.cat(all_preds).detach().numpy()
    all_targets = torch.cat(all_targets).detach().numpy()
    
    # Calculate the Spearman correlation for each cell type 
    # Press spot
    spot_scores = [
        spearmanr(all_preds[j], all_targets[j]).correlation
        for j in range(all_preds.shape[0])
    ]
    spot_avg = np.nanmean(spot_scores)
    
    avg_epoch_loss = total_loss / len(dataloader.dataset)
    return avg_epoch_loss, spot_avg

from scipy.stats import rankdata

def evaluate(model, dataloader, device, current_epoch,
             initial_alpha=0.3, final_alpha=0.8, target_epoch=15, method="linear", loss_type = 'pearson'):
    """
    Evaluation function:The hybrid loss is calculated using the same dynamic alpha as the train function.
    Only keep necessary parameters:
    
    :return: Average loss, Average Spearman, Each cell type's avg MSE and Spearman 
    """
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    total_mse = torch.zeros(35).to(device)  # Assuming there are 35 cell types
    n_samples = 0
    
    # Based on the current epoch calculate dynamic alpha
    alpha = get_alpha(current_epoch, initial_alpha, final_alpha, target_epoch, method)
    
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            inputs, label = make_input_to_device(model, batch, device)
            out = model(**inputs)
            loss = hybrid_loss(out, label, alpha=alpha, loss_type=loss_type)
            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            preds.append(out.cpu())
            targets.append(label.cpu())
            loss_per_cell = ((out - label) ** 2).sum(dim=0)  # (35,)
            total_mse += loss_per_cell
            n_samples += batch_size
            pbar.set_postfix(loss=loss.item())
    
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mse_per_cell = (total_mse / n_samples).cpu().numpy()

    spot_scores = [
            spearmanr(preds[j], targets[j]).correlation
            for j in range(preds.shape[0])
        ]
    spearman_spot_avg = np.nanmean(spot_scores)
    
    # First,sort the matrix by row（axis=1）and perform a relative sorting of the 35 dimensions within the spot
    preds_ranked   = np.apply_along_axis(lambda r: rankdata(r, method="ordinal"), 1, preds)
    targets_ranked = np.apply_along_axis(lambda r: rankdata(r, method="ordinal"), 1, targets)

     # Cell-type-level Spearman（Calculate Spearman dirrectly for each column）
    spearman_per_cell = []
    for j in range(preds.shape[1]):
        corr = spearmanr(preds[:, j], targets[:, j]).correlation
        spearman_per_cell.append(corr)
    
    avg_epoch_loss = total_loss / n_samples
    return avg_epoch_loss, spearman_spot_avg, mse_per_cell, spearman_per_cell


__all__ = [
    "get_model_inputs",
    "train_one_epoch",
    "evaluate",
    "predict",
    "EarlyStopping",
    "plot_losses"
]