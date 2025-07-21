import os
def dataset_statistic(data_dir):
    """
    数据集的统计信息
    
    Args:
        data_dir: 数据集根目录
        
    Returns:
        all_image_paths: 所有图像路径列表
        all_labels: 所有图像标签列表
        class_names: 类别名称列表
    """
    # 定义类别
    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    
    # 收集所有图像路径和对应标签
    all_image_paths = []
    all_labels = []
    all_lengths = []
    
    for cls_name in class_names:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
            
        cls_idx = class_to_idx[cls_name]
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(img_path)
                all_labels.append(cls_idx)
        all_lengths.append(len(os.listdir(cls_dir)))
        
        
    # 打印统计信息
    print("Dataset statistics:")
    print("Number of classes: {}".format(len(class_names)))
    print("Number of images: {}".format(len(all_image_paths)))
    print("Number of images per class:")
    for cls_name, length in zip(class_names, all_lengths):
        print("  {}: {}".format(cls_name, length))
        
    return all_image_paths, all_labels, class_names

if __name__ == "__main__":
    data_dir = "garbage-dataset"
    all_image_paths, all_labels, class_names = dataset_statistic(data_dir)