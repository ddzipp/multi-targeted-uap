import json

import timm
import torch
import torchvision
from accelerate import Accelerator
from tqdm import tqdm

# 初始化accelerator
accelerator = Accelerator()


# 加载预训练的EVA-02 Large模型
model = timm.create_model("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", pretrained=True)

data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_cfg)
model.eval()  # 设置为评估模式

test_imagenet_path = "./data/ImageNet/test"

batch_size = 128
test_dataset = torchvision.datasets.ImageFolder(test_imagenet_path, transform=transform)
# test_dataset = torch.utils.data.Subset(test_dataset, range(10))  # 只使用前1000张图像
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 使用accelerator准备模型和数据加载器
model, dataloader = accelerator.prepare(model, dataloader)

# 创建一个字典来存储预测结果
predictions = {}

with torch.no_grad():
    # 遍历测试数据集并进行预测
    for idx, (image, _) in enumerate(tqdm(dataloader)):
        # 进行预测
        output = model(image)
        # 确保在CPU上进行后处理
        preds = output.argmax(-1).cpu().numpy()

        # 获取当前批次中的图像文件名
        for i in range(len(preds)):
            # 计算正确的全局索引，考虑进程索引
            global_idx = idx * batch_size * accelerator.num_processes + accelerator.process_index * batch_size + i
            if global_idx < len(test_dataset):
                img_path = test_dataset.imgs[global_idx][0]
                # img_path = test_dataset.dataset.imgs[global_idx][0]
                img_name = img_path.split("/")[-1]
                predictions[img_name] = int(preds[i])

        # # 每10个批次保存一次结果
        # if (idx + 1) % 100 == 0:
        #     # 收集所有进程的当前预测结果
        #     # all_predictions是一个列表，包含每个进程的predictions字典
        #     all_predictions = accelerator.gather(predictions)  # List[Dict[str, int]]

        #     if accelerator.is_main_process:
        #         # 合并所有进程的预测结果
        #         checkpoint_predictions = {}
        #         for pred_dict in all_predictions:  # 遍历每个进程的预测结果
        #             checkpoint_predictions.update(pred_dict)

        #         print(f"保存第{idx+1}批次的预测结果...")
        #         with open(
        #             f"./save/imagenet_test_labels_{idx}.json", "w", encoding="utf-8"
        #         ) as f:
        #             json.dump(checkpoint_predictions, f)

    # 收集所有进程的最终预测结果
    # all_predictions是一个列表，每个元素是一个进程的predictions字典
    # 保存每个进程的预测结果
    with open(
        f"./save/imagenet_test_labels_{accelerator.process_index}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(predictions, f)

"""

def process_imagenet_test_labels(file_path):
    import json
    from collections import defaultdict
    import seaborn as sns
    import matplotlib.pyplot as plt

    json_list = []
    file_list = [f"./save/imagenet_test_labels_{i}.json" for i in range(7)]
    for file in file_list:
        with open(file, "r") as f:
            json_list.append(json.load(f))

    total_json = {}
    for j in json_list:
        for key, value in j.items():
            total_json[key] = value

    with open("./save/imagenet_test_labels.json", "w") as f:
        json.dump(total_json, f, indent=4, sort_keys=True)

    label2index = defaultdict(list)
    for key, value in total_json.items():
        label2index[value].append(key)

    # sort label2index each list
    for key, value in label2index.items():
        label2index[key] = sorted(value)

    # save label2index
    with open("./save/imagenet_test_label2index.json", "w") as f:
        json.dump(label2index, f, indent=4, sort_keys=True)


    label2index = dict(sorted(label2index.items()))
    image_num = []
    for i in label2index.keys():
        image_num.append(len(label2index[i]))
    
    print("min: ", min(image_num), "max: ", max(image_num))
    
    plt.figure(figsize=(12, 6))
    # 绘制直方图
    sns.histplot(data=image_num, bins=50, color='indianred',stat='density', alpha=0.7)

    # 添加曲线
    sns.kdeplot(data=image_num, color='navy', linewidth=2)

    plt.title('The number of images in each class', fontsize=14)
    plt.xlabel('The number of images', fontsize=12)
    plt.ylabel('Density of classes', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    
"""
