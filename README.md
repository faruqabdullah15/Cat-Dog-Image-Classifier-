Project Overview
This is a 4-class image classification project that classifies images into:
- "both" - Image contains both cat and dog
- "cat" - Image contains only cat
- "dog" - Image contains only dog  
- "neither" - Image contains neither cat nor dog

The project uses **Deep Learning** with **Transfer Learning** approach.

- Algorithm: Convolutional Neural Network (CNN) with Residual Blocks
- Architecture: ResNet18 (18 layers deep)

ResNet Architecture Details:

Input Image (224x224x3)
    ↓
Convolutional Layers (Feature Extraction)
    ↓
Residual Blocks (with skip connections)
    ↓
Global Average Pooling
    ↓
Fully Connected Layer (512 → 4 classes)
    ↓
Output: 4 probabilities [both, cat, dog, neither]

NOTE: I am not providing the dataset, you can get it by using kaggle and also this code works 75-80% (sometimes detects normal picture as a cat/dog). So if there is a correction in logic or code let me know. Thanks in advance.

