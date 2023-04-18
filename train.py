import matplotlib.image as mpi
from torchvision.models import resnet50, ResNet50_Weights
import torch
import numpy as np


if __name__ == "__main__":
    img = mpi.imread("img/view1_frontal.jpg")  # grayscale image shape (d1, d2)
    img = np.expand_dims(img, axis=0)  # so that it follows format (N, C, d1, d2)
    img = np.repeat(img, repeats=3, axis=0)  # simulate RGB
    img = torch.tensor(img)

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"the end: {category_name}: {100 * score:.1f}%")
