import numpy as np


IMAGES_PREFIX = "/groups/CS156b/2023/Xray-diagnosis/data/"


if __name__ == "__main__":
    inputs_part1 = np.load(IMAGES_PREFIX + "part1_allTrain_224x224.npy")
    inputs_part2 = np.load(IMAGES_PREFIX + "part2_allTrain_224x224.npy")
    inputs_part3 = np.load(IMAGES_PREFIX + "part3_allTrain_224x224.npy")
    assert inputs_part1.shape == (60000, 224, 224)
    assert inputs_part2.shape == (59079, 224, 224)
    assert inputs_part3.shape == (59078, 224, 224)
    inputs = np.concatenate((inputs_part1, inputs_part2, inputs_part3), axis=0, dtype=np.int8)
    with open(f"/groups/CS156b/2023/Xray-diagnosis/data/allTrain_224x224.npy", "wb") as f:
        np.save(f, inputs)
