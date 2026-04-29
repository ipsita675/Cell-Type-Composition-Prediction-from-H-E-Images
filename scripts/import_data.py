import torch
from torch.utils.data import Dataset
from .operate_model import get_model_inputs
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

def preprocess_data(data, image_keys, transform):
    """
    The 'image_keys' field in the specified 'data' dictionary is preprocessed using the 'transform' property.
    If each entry in a field is a single image,convert one by one
    If each piece of data is a list(e.g., subtiles or neighbor_tiles), then each image in that list is transformed.
    Parameter:
      data: A dictionary of original data, in the format{key: list([...])},
            The key may contain images(e.g., 'center_tile', 'subtiles', ...）。
      image_keys: list or set specifies which fields contain image data that need to be preprocessed
      transform: A torchvision transformation function, such as transforms.Compose([...])
      
    Return:
      processed_data:A new data dictionary where the fields in image_keys have been transformed.
                      The other files remain unchanged
    """
    processed_data = {}
    for key, value in data.items():
        if key in image_keys:
            # Check if the first record in this field is a list
            if isinstance(value[0], list):
                # Process each image in each piece of data
                processed_data[key] = [
                    [transform(img) for img in sublist] for sublist in value
                ]
            else:
                # Single image processing
                processed_data[key] = [transform(img) for img in value]
        else:
            # Non image fields remain
            processed_data[key] = value
    return processed_data


if __name__ == "__main__":
    # Conversion process for images
    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Assuming the original image uses PIL.Image
    dummy_img = Image.new("RGB", (64, 64))  # 建立一個 64x64 的空白 RGB 圖片

    num_samples = 100
    original_data = {
        'center_tile': [dummy_img for _ in range(num_samples)],
        'subtiles': [[dummy_img for _ in range(9)] for _ in range(num_samples)],
        'neighbor_tiles': [[dummy_img for _ in range(8)] for _ in range(num_samples)],
        'coords': [[0.5, 0.5] for _ in range(num_samples)],
        'label': [torch.randn(35, dtype=torch.float32) for _ in range(num_samples)]
    }

    image_keys = ['center_tile', 'subtiles', 'neighbor_tiles']

    # Perform preprocessing on the specified image field 
    processed_data = preprocess_data(original_data, image_keys, my_transform)

    



import torch
from torch.utils.data import Dataset
import inspect
import numpy as np

def convert_item(item, is_image=False):
    """
    Convert any list / numpy.ndarray / Python scalar to torch.Tensor。
    If is_image=True and item is an ndarray,then perform a transform of HxWx3 → 3xHxW on the NumPy side,then
    directly convert it to a Tensor;otherwise, recursively convert the pure Python structure to a Tensor。
    """
    # 1) If it's a NumPy array of images,directly perform channel-last → channel-first on the Numpy side
    if is_image and isinstance(item, np.ndarray):
        arr = item.astype(np.float32)
        if arr.ndim == 3 and arr.shape[2] == 3:
            # [H, W, 3] → [3, H, W]
            arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 4 and arr.shape[-1] == 3:
            # [B, H, W, 3] → [B, 3, H, W]
            arr = arr.transpose(0, 3, 1, 2)
        return torch.from_numpy(arr)

    # 2) All other NumPy arrays should first be reduced to Python list
    if isinstance(item, np.ndarray):
        try:
            item = item.tolist()
        except Exception:
            # If tolist() fails，first force it to float32 and then reduce it to list
            item = np.asarray(item, dtype=np.float32).tolist()

    # 3) If it's a list，recursively convert and then stack
    if isinstance(item, list):
        converted = [convert_item(elem, is_image=is_image) for elem in item]
        try:
            return torch.stack(converted)
        except Exception:
            raise ValueError(f"Failed to convert elements in list, list content: {item}")

    # 4) If it is already a Tensor，return directly
    if isinstance(item, torch.Tensor):
        return item

    # 5) The rest are Python scalars（int/float/...），which can be directly used with torch.tensor()
    try:
        return torch.tensor(item)
    except Exception:
        raise ValueError(f"Unable to convert data to tensor,Input data: {item}")


class importDataset(Dataset):
    def __init__(self, data_dict, model, image_keys=None, transform=None, print_sig=False):
        self.data = data_dict
        self.image_keys = set(image_keys) if image_keys is not None else set()
        self.transform = transform if transform is not None else lambda x: x
        self.forward_keys = list(get_model_inputs(model, print_sig=print_sig).parameters.keys())

        expected_length = None
        for key, value in self.data.items():
            if expected_length is None:
                expected_length = len(value)
            if len(value) != expected_length:
                raise ValueError(f"The length ({len(value)}) of the data field '{key}' is inconsistent with the expected length({expected_length}) ")

        for key in self.forward_keys:
            if key not in self.data:
                raise ValueError(f"data_dict is a missing model forward.Required field '{key}'.Currently availaible fields: {list(self.data.keys())}")
        if "label" not in self.data:
            raise ValueError(f"data_dict must contain a 'label' field. Availaible Fields: {list(self.data.keys())}")
        if "source_idx" not in self.data:
            raise ValueError("The data_dict must contain a 'source_idx' field to trace the original order")
        if "position" not in self.data:
            raise ValueError("The data_dict must contain a 'position' field to trace the original order")
    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        sample = {}
        for key in self.forward_keys:
            value = self.data[key][idx]
            value = self.transform(value)
            value = convert_item(value, is_image=(key in self.image_keys))
            if isinstance(value, torch.Tensor):
                value = value.float()
            sample[key] = value

        label = self.transform(self.data["label"][idx])
        label = convert_item(label, is_image=False)
        if isinstance(label, torch.Tensor):
            label = label.float()
        sample["label"] = label

        # Add source_idx
        source_idx = self.data["source_idx"][idx]
        sample["source_idx"] = torch.tensor(source_idx, dtype=torch.long)
        # Add position （Assuming 'position' in data_dict as (x, y) or [x, y]）
        pos = self.data["position"][idx]
        sample["position"] = torch.tensor(pos, dtype=torch.float)
        return sample
    def check_item(self, idx=0, num_lines=5):
        expected_keys = self.forward_keys + ['label', 'source_idx', 'position']
        sample = self[idx]
        print(f"🔍 Checking dataset sample: {idx}")
        for key in expected_keys:
            if key not in sample:
                print(f" The data is missing in key: {key}")
                continue
            tensor = sample[key]
            if isinstance(tensor, torch.Tensor):
                try:
                    shape = tensor.shape
                except Exception:
                    shape = "N/A"
                dtype = tensor.dtype if hasattr(tensor, "dtype") else "N/A"
                output_str = f"📏 {key} shape: {shape} | dtype: {dtype}"
                if tensor.numel() > 0:
                    try:
                        tensor_float = tensor.float()
                        mn = tensor_float.min().item()
                        mx = tensor_float.max().item()
                        mean = tensor_float.mean().item()
                        std = tensor_float.std().item()
                        output_str += f" | min: {mn:.3f}, max: {mx:.3f}, mean: {mean:.3f}, std: {std:.3f}"
                    except Exception:
                        output_str += "| Unable to calculate statistics"
                print(output_str)
                if key not in self.image_keys:
                    if tensor.ndim == 0:
                        print(f"--- {key} data is a scalar:", tensor)
                    elif tensor.ndim == 1:
                        print(f"--- {key} head (first {num_lines} elements):")
                        print(tensor[:num_lines])
                    else:
                        print(f"--- {key} head (previous {num_lines} column):")
                        print(tensor[:num_lines])
            else:
                # If position stores a list/tuple/etc，this will also be called:
                print(f" {key} (non-tensor data):", tensor)
        print(" All checks passed!")



import os
import torch
import random

def load_all_tile_data(folder_path,
                       model,
                       fraction: float = 1.0,
                       shuffle : bool = False):
    """
    Return a dict,which contains：
        - The fields required by Model forward() 
        - 'label'
        - 'slide_idx'    ← for GroupKFold
        - 'source_idx'   ← Read from inside the .pt file
    """
    sig            = get_model_inputs(model, print_sig=False)
    fwd_keys       = list(sig.parameters.keys())
    required_keys  = set(fwd_keys + ['label', 'slide_idx', 'position'])   # include slide_idx
    keep_meta_keys = required_keys.union({'source_idx'})

    pt_files = sorted(f for f in os.listdir(folder_path) if f.endswith('.pt'))
    N        = len(pt_files)
    keep_n   = max(1, int(N * fraction))
    pt_files = random.sample(pt_files, keep_n) if shuffle else pt_files[-keep_n:]

    data_dict = {k: [] for k in keep_meta_keys}

    for fname in pt_files:
        fpath = os.path.join(folder_path, fname)
        d = torch.load(fpath, map_location='cpu',weights_only = False)

        #  Read source_idx from within the file first.
        if 'source_idx' in d:
            data_dict['source_idx'].append(d['source_idx'])
        else:
            data_dict['source_idx'].append(fname)  # optional fallback

        # Add other fields
        for k in required_keys:
            data_dict[k].append(d.get(k, None))

    return data_dict




def load_node_feature_data(pt_path: str, model, num_cells: int = 35) -> dict:
    """
    Based on the parameters of model.forward automatically load the required fields from the .pt file,
    It will automatically fill in the 'label'(if it doesn't exist)as a 0 tensor。
    Supports automatic reading of the 'position' and 'source_idx' fields(if used by forward)

    Return:
      dict: key corresponds to the parameter name of the forward method + label, position, source_idx(if needed)
    """
    import torch
    import inspect

    raw = torch.load(pt_path, map_location="cpu",weights_only=False)

    # What params are needed for a horizontal layout？
    sig = inspect.signature(model.forward)
    param_names = [p for p in sig.parameters if p != "self"]
    param_names.append('source_idx')
    param_names.append('position')

    out = {}
    for name in param_names:
        # a) Direct same name
        if name in raw:
            out[name] = raw[name]
            continue
        # b) name + 's'（plural）
        if name + "s" in raw:
            out[name] = raw[name + "s"]
            continue
        # c) Fuzzy matching
        cands = [k for k in raw if name in k or k in name]
        if len(cands) == 1:
            out[name] = raw[cands[0]]
            continue
        raise KeyError(f"Cannot find '{name}',raw keys: {list(raw.keys())}")

    # Infer batch size
    dataset_size = None
    for v in out.values():
        if hasattr(v, "__len__"):
            dataset_size = len(v)
            print(f"Inferring the sample size from'{type(v)}': {dataset_size}")
            break
    if dataset_size is None:
        raise RuntimeError("Unable to infer the number of samples")

    # Add label
    out["label"] = raw.get("label", torch.zeros((dataset_size, num_cells), dtype=torch.float32))

    # Add position and source_idx（If applicable）
    for meta_key in ["position", "source_idx"]:
        if meta_key in raw:
            out[meta_key] = raw[meta_key]

    return out


if __name__ == "__main__":
    # Define a model，assuming the forward function requires the following parameters: center_tile, subtiles, neighbor_tiles, coords
    class DummyModel:
        def forward(self, center_tile, subtiles, neighbor_tiles, coords):
            pass

    model = DummyModel()

    # Simulate 100 data entries，with each data entry's original image format being channel-first (3, H, W)
    num_samples = 100
    dummy_center = [torch.randn(3, 64, 64) for _ in range(num_samples)]
    dummy_subtiles = [[torch.randn(3, 32, 32) for _ in range(9)] for _ in range(num_samples)]
    dummy_neighbor = [[torch.randn(3, 64, 64) for _ in range(8)] for _ in range(num_samples)]
    dummy_coords = [[0.5, 0.5] for _ in range(num_samples)]
    dummy_label = [torch.randn(35, dtype=torch.float32) for _ in range(num_samples)]  # 假設 label 長度為 35

    # Create a data dictionary，the key names must watch DummyModel.forward and include labels.
    data = {
        'center_tile': dummy_center,
        'subtiles': dummy_subtiles,
        'neighbor_tiles': dummy_neighbor,
        'coords': dummy_coords,
        'label': dummy_label
    }

    # Specify which fields are for image data
    image_keys = ['center_tile', 'subtiles', 'neighbor_tiles']

    # Create a ValidatedDataset
    dataset = importDataset(
        data_dict=data,
        model=model,
        image_keys=image_keys,
        transform=lambda x: x,  # identity transform
        print_sig=True
    )

    # Get the first information
    sample = dataset[0]
    # Print the combined data in the following order:(center_tile, subtiles, neighbor_tiles, coords, label)
    print("Order of the obtained sample data:")
    print("center_tile shape:", sample[0].shape)
    print("subtiles shape:", sample[1].shape)
    print("neighbor_tiles shape:", sample[2].shape)
    print("coords:", sample[3])
    print("label shape:", sample[4].shape)

    # Check the first data
    dataset.check_item(idx=0)
    