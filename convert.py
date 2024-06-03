from ultralytics import YOLO
import numpy as np
# Load a model
model = YOLO("/mnt/d/projects/ultralytics/ultralytics/yolov8.yaml")  # build a new model from scratch
model = YOLO("/mnt/d/projects/ultralytics/runs/detect/train2/weights/best.pt")  # load a pretrained model (recommended for training)
original_image: np.ndarray = np.load("/mnt/d/projects/datasets/dataset_root/potato_data/yolo_depth/images/val/0083.npz")
original_image = original_image["array"]
# crop_height, crop_width = 640, 640
# start_y = (original_image.shape[0] - crop_height) // 2
# start_x = (original_image.shape[1] - crop_width) // 2
# original_image = original_image[start_y:start_y + crop_height, start_x:start_x + crop_width]
current_min = np.min(original_image[np.nonzero(original_image)])
current_max = np.max(original_image[np.nonzero(original_image)])
mask = original_image !=0
original_image = np.where(mask ,((original_image - current_min) / (current_max - current_min)) * (1 - 0) + 0, original_image)
original_image = original_image*255
original_image = np.expand_dims(original_image,axis=0)
original_image = original_image.transpose((1,2,0))
results = model(original_image)
print(results)
# path = model.export(format="onnx")  # export the model to ONNX format