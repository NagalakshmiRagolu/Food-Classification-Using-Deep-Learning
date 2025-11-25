🍽️ Food Image Classification Using CNNs with Nutritional Analysis
1. Project Overview

Food classification using deep learning is an emerging domain in computer vision.
The goal of this project is to automatically:

Identify food items from images

Provide nutritional information

Assist in calorie tracking and diet analysis

This system is useful for:

Diet tracking & calorie monitoring

Healthcare diet supervision

Food recommendation systems

Smart kitchens & IoT devices

The project includes three CNN models:

VGG-16 – deep architecture with strong accuracy

ResNet-50 – uses residual blocks, avoids vanishing gradients

Custom CNN – lightweight architecture for real-time inference

The pipeline includes preprocessing, augmentation, training, model evaluation, and JSON result generation.

2. Dataset Description

The dataset contains 40+ food categories, each having 200+ images in JPG format.

✔ Sample Dataset Table (GitHub Compatible)
Food Name	Images	Calories	Protein	Fat	Fiber
Apple Pie	200	531	22g	15g	8g
Baked Potato	200	338	25g	22g	1g
Burger	200	258	17g	2g	1g
Butter Naan	200	530	1g	7g	8g
Chai	200	511	5g	17g	10g
…	…	…	…	…	…

Nutritional details are stored in:

📁 food_nutrition.json
Contains: calories, proteins, fats, carbs, and fiber for every food class.

3. Model Architecture
3.1 VGG-16

16 layers (13 Conv + 3 FC)

Input size: 224×224×3

Pretrained on ImageNet

Fine-tuned for food dataset

3.2 ResNet-50

50-layer deep network

Uses residual blocks

High performance on complex food images

3.3 Custom CNN

4 convolutional layers

MaxPooling layers

Lightweight softmax classifier

Fastest inference among all three models

4. Data Preprocessing

Resize images → 224×224

Normalize pixel values → 0–1

Augmentation:

Rotation

Horizontal flip

Zoom

Shear

Dataset Splitting:

Train → 70%

Validation → 15%

Test → 15%

Classes were grouped into:

➡️ Group_1 … Group_11
for modular training and evaluation.

5. Training & Evaluation

Each model is evaluated on:

Accuracy

Precision

Recall

F1-score

TP, FP, FN, TN

All results are stored in:

📁 model_performance.json

6. Results & Analysis
✔ Sample JSON Output (Clean Formatted)
{
  "Model_Name": "VGG-16",
  "Model_File": "vgg16_group_1.h5",
  "Group_Name": "Group_1",
  "Test_Samples": 180,
  "Test_Accuracy": 88.75,
  "Classes": [
    {
      "Class_Name": "Biryani",
      "Support": 30,
      "Precision(%)": 90.0,
      "Recall(%)": 93.33,
      "F1(%)": 91.62,
      "TP": 28,
      "FP": 3,
      "FN": 2,
      "TN": 147
    }
  ]
}

✔ Confusion Matrix Example (GitHub Table)
Actual \ Predicted	Apple Pie	Baked Potato	Burger
Apple Pie	28	1	1
Baked Potato	2	27	1
Burger	1	2	27
7. Nutritional Data Analysis

Using food_nutrition.json, you can generate:

Calorie distribution graphs

Protein comparison graphs

Fiber variation charts

Fat content analysis

Useful for:

✔ Meal planning
✔ Diet optimization
✔ Calorie tracking

8. Future Scope

Real-time prediction using mobile camera

Add more Indian & international cuisines

Portion size detection

Explainable AI (Grad-CAM Heatmaps)

Cloud deployment (AWS/GCP/Render)

Integration with fitness apps

9. References

Simonyan, K., Zisserman, A. (2014) — Very Deep CNNs for Image Recognition

He, K., Zhang, X., Ren, S., Sun, J. (2016) — Deep Residual Networks

Food-101 Dataset: https://www.vision.ee.ethz.ch/datasets_extra/food-101/

📌 Project Author Details
Field	Details
Name	Nagalakshmi Ragolu
Email	nagalakshmiragolu@gmail.com

LinkedIn	https://www.linkedin.com/in/ragolu-nagalakshmi-71587a22

Education	B.Tech – Information Technology
