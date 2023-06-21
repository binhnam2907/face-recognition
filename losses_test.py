import tensorflow as tf
import numpy as np

def arcface_loss(feature1, feature2, margin=0.5, scale=64):
    feature1_norm = feature1 / np.linalg.norm(feature1)
    feature2_norm = feature2 / np.linalg.norm(feature2)
    similarity = np.dot(feature1_norm, feature2_norm.T)
    theta = np.arccos(similarity)
    target = np.cos(theta + margin)
    logits = scale * target
    loss = -np.log(np.exp(logits) / (np.exp(logits) + np.exp(-logits)))

    return loss

if __name__ == '__main__':
    feature1 = np.random.randn(1, 512)
    feature2 = np.random.randn(1, 512)

    loss = arcface_loss(feature1, feature2)
    print(loss)

