from typing import Any, Dict, Optional
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# 注意：不再需要 VisionData dataclass，因为 __getitem__ 直接返回字典

# 定义驾驶场景数据集类，继承自 PyTorch 的 Dataset
class DrivingSceneDataset(Dataset):
    def __init__(
            self,
            root_path: str,  # 数据集的根目录路径
            scene_id: int,  # 要加载的场景的ID (0索引)
            target_mapping: Dict[int, str],  # 细分场景标签(0-4)到目标文本的映射字典
            transform: Optional[torchvision.transforms.Compose] = None,  # 可选的图像预处理转换
            scenes_per_dataset: int = 15,  # 数据集中预期的总场景文件夹数量 (用于校验)
            images_per_scene: int = 50,  # 每个场景中预期的图片数量
            images_per_segment: int = 10,  # 每个细分驾驶场景包含的图片数量
    ):

        super().__init__()
        self.root_path = root_path
        self.scene_id = scene_id
        self.target_mapping = target_mapping
        # 如果没有提供 transform，则默认使用 ToTensor()
        self.transform = transform if transform is not None else transforms.ToTensor()

        self.images_per_scene = images_per_scene
        self.images_per_segment = images_per_segment
        # 计算每个场景有多少个细分驾驶场景 (segments)
        self.num_segments = images_per_scene // images_per_segment

        if not (self.num_segments == 5 and images_per_scene % images_per_segment == 0):
            print(f"警告: 计算得到的细分场景数量是 {self.num_segments}。请确保这符合您的预期（标签0-4意味着5个细分场景）。")

        # 发现所有场景文件夹并选择目标场景
        all_scene_folders = sorted([
            d for d in os.listdir(self.root_path)
            if os.path.isdir(os.path.join(self.root_path, d)) and d.startswith("Scene")
        ])

        if not (0 <= scene_id < len(all_scene_folders)):
            raise ValueError(
                f"场景ID {scene_id} 超出范围。共找到 {len(all_scene_folders)} 个场景。"
            )
        if len(all_scene_folders) != scenes_per_dataset:
            print(
                f"警告: 预期有 {scenes_per_dataset} 个场景文件夹，但在 {self.root_path} 中找到了 {len(all_scene_folders)} 个。")

        self.scene_folder_name = all_scene_folders[scene_id]
        self.scene_path = os.path.join(self.root_path, self.scene_folder_name)

        # 加载选定场景的图像文件路径 (文件名从 "0001.jpg" 开始)
        # glob.glob 返回的是无序的，所以需要排序
        self.image_files = sorted(glob.glob(os.path.join(self.scene_path, "*.jpg")))

        if len(self.image_files) != self.images_per_scene:
            raise ValueError(
                f"场景 '{self.scene_folder_name}' 中预期有 {self.images_per_scene} 张图像，但找到了 {len(self.image_files)} 张。"
            )

        # 校验 target_mapping 中的键是否覆盖了所有细分场景标签
        expected_labels = set(range(self.num_segments))
        if set(self.target_mapping.keys()) != expected_labels:
            print(
                f"警告: 场景 {self.scene_folder_name} 的 target_mapping 中的键 {set(self.target_mapping.keys())} "
                f"与预期的细分场景标签 {expected_labels} 不匹配。"
            )

    def __len__(self) -> int:
        """返回当前场景中的图像数量。"""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # 返回值类型修改为字典
        """
        从数据集中检索一个数据项。

        参数:
            idx (int): 当前场景中图像的0索引 (从 0 到 images_per_scene - 1)。
                       例如，idx=0 对应文件 "0001.jpg", idx=9 对应 "0010.jpg"。

        返回:
            Dict[str, Any]: 一个包含图像、标签、问题、答案和目标文本的字典。
        """
        if not (0 <= idx < len(self.image_files)):
            raise IndexError(f"索引 {idx} 超出当前场景图像数量 {len(self.image_files)} 的范围。")

        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"打开图像文件 {img_path} 时出错: {e}")

        if self.transform:
            image = self.transform(image)

        # 索引 0-9 (对应文件 0001-0010) 属于标签 0
        # 索引 10-19 (对应文件 0011-0020) 属于标签 1
        # ...
        # segment_label = idx // self.images_per_segment 仍然适用，
        # 因为 idx 是0-based，而 self.images_per_segment 是10。
        segment_label = idx // self.images_per_segment

        if not (0 <= segment_label < self.num_segments):  # 校验标签范围
            raise ValueError(
                f"为图像索引 {idx} 计算得到的细分场景标签 {segment_label} 超出预期范围 [0, {self.num_segments - 1}]。")

        target_text = self.target_mapping.get(segment_label)
        if target_text is None:
            print(
                f"警告: 在场景 {self.scene_folder_name} 中未找到细分场景标签 {segment_label} 对应的目标文本。将使用默认值。")
            target_text = "默认目标文本 - 映射缺失。"

        response = {
            "image": image,
            "label": str(segment_label),  # 将标签存储为字符串，或直接用整数 segment_label
            "question": "Describe this image.",
            "answer": "Unknown.",
            "target": target_text,
        }
        return response

    def get_scene_name(self) -> str:
        """返回当前加载的场景的文件夹名称。"""
        return self.scene_folder_name
