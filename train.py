from ultralytics import YOLO
import numpy as np
# Load a model
model = YOLO("/mnt/d/projects/ultralytics/ultralytics/yolov8.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/mnt/d/projects/datasets/dataset_root/potato_data/yolo_depth/potato.yaml", epochs=500)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
f = "/mnt/d/projects/datasets/dataset_root/potato_data/yolo_depth/images/val/0083.npz"
im = np.load(f)
im = im["array"]
current_min = np.min(im[np.nonzero(im)])
current_max = np.max(im[np.nonzero(im)])
mask = im !=0
im = np.where(mask,((im - current_min) / (current_max - current_min)) * (1 - 0) + 0, im)
im = im*255
im = np.expand_dims(im,axis=0)
im = im.transpose((1,2,0))
results = model(im)  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format