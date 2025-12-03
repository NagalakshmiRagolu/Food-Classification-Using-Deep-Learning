#🍽️ Food Image Classification Using CNNs with Nutritional Analysis
A deep learning project that classifies food images into 34 categories and automatically provides nutritional information (calories, protein, fat, carbohydrates, and fiber) for each predicted dish. This can support diet tracking, healthcare diet supervision, food recommendation systems, and smart kitchen/IoT applications.​

📌 Table of Contents
Project Overview

Key Features

Food Classes (34 Categories)

Concepts & Terminology (Clearly Defined)

Dataset & Nutritional Annotations

Data Preprocessing & Augmentation

Model Architectures (VGG‑16, ResNet‑50, Custom CNN)

Training Setup & Metrics

Nutritional Analysis Pipeline

Tech Stack & Library Explanations

Project Structure

How to Run

Future Work

References

Author

🧾 Project Overview
The system takes a food image as input, predicts which dish it is using Convolutional Neural Networks (CNNs), and then looks up its nutritional profile from a JSON file. The final output is a structured JSON object containing the predicted class, confidence score, and macro-nutrients per serving for that dish.​

⭐ Key Features
34-class food image classification with 200+ images per class.​

Three CNN models:

VGG‑16 (pretrained, fine-tuned)

ResNet‑50 (deep residual model)

Custom CNN (lightweight, fast inference)​

Central nutritional database in food_nutrition.json (calories, protein, fat, carbs, fiber for each class).​

Evaluation with accuracy, precision, recall, F1‑score, TP, FP, FN, TN and confusion matrices, saved in model_performance.json.​

JSON-based prediction output, easy to integrate into mobile apps/APIs.

📋 Food Classes (34 Categories)
Each class is a folder name in the dataset and an entry in food_nutrition.json.

S.No	Class Name	Description (Short)
1	apple_pie	Baked dessert with sweet apple filling
2	Baked Potato	Oven-baked whole potato
3	burger	Patty in a sliced bun with toppings
4	butter_naan	Leavened Indian flatbread with butter
5	chai	Spiced Indian milk tea
6	chapati	Unleavened whole wheat flatbread
7	cheesecake	Cream cheese-based baked or chilled dessert
8	chicken_curry	Chicken pieces cooked in spiced gravy
9	chole_bhature	Spiced chickpeas with deep-fried bread
10	Crispy Chicken	Deep-fried seasoned chicken
11	dal_makhani	Creamy lentil curry made with black lentils
12	dhokla	Steamed fermented gram flour cake
13	Donut	Deep-fried sweet dough ring or filled piece
14	fried_rice	Stir-fried rice with vegetables or eggs
15	Fries	Deep-fried potato strips
16	Hot Dog	Sausage in a sliced bun
17	ice_cream	Frozen sweet dairy or non-dairy dessert
18	idli	Steamed rice-lentil cakes
19	jalebi	Deep-fried spiral sweet soaked in syrup
20	kaathi_rolls	Stuffed wrap made with roti or paratha
21	kadai_paneer	Cottage cheese cooked in spiced tomato gravy
22	kulfi	Dense traditional Indian frozen dessert
23	masala_dosa	Dosa filled with spiced potato mixture
24	momos	Steamed or fried dumplings with filling
25	omelette	Beaten eggs cooked flat, sometimes with veg
26	paani_puri	Hollow puris filled with spicy tangy water
27	pakode	Deep-fried vegetable or paneer fritters
28	pav_bhaji	Spiced mashed vegetable curry with bread
29	pizza	Flatbread topped with sauce, cheese, extras
30	samosa	Fried pastry filled with spiced potatoes etc
31	Sandwich	Filling between slices of bread
32	sushi	Vinegared rice with fillings or toppings
33	Taco	Folded tortilla with savory filling
34	Taquito	Rolled tortilla filled and fried or baked
📚 Concepts & Terminology
Image Classification: Assigning a single label (here, a food class) to an entire input image.

CNN (Convolutional Neural Network): Neural network using convolution layers to learn patterns (edges, textures, shapes) from images automatically.​

Pretrained Model: A model (e.g., VGG‑16, ResNet‑50) already trained on a large dataset like ImageNet, reused and fine‑tuned on this food dataset.​

Fine‑tuning: Unfreezing some layers of a pretrained model and continuing training on a new dataset to adapt it to the new task.

Residual Block / Skip Connection: A structure where the input is added to the output of some layers (used in ResNet‑50) to make very deep networks trainable.​

Overfitting: When a model learns training data too specifically and performs poorly on new data; augmentation and regularization reduce this.​

Confusion Matrix: A table that compares true labels vs predicted labels to show TP, FP, FN, and TN for each class.

Precision / Recall / F1‑score:

Precision: Of all samples predicted as a class, how many are correct.

Recall: Of all true samples of a class, how many are found.

F1: Harmonic mean of precision and recall, balancing both.

🧂 Dataset & Nutritional Annotations
Number of classes: 34.

Images per class: ≥ 200 JPG images.

Format: RGB images, resized to 224×224 for model input.

Folder structure: Each class has its own folder (e.g., data/train/apple_pie, data/train/burger).​

Nutritional Data (food_nutrition.json)
food_nutrition.json stores macro‑nutrient values for each food class:

calories – Energy per serving (kcal).

protein_g – Protein content in grams.

fat_g – Fat content in grams.

carbs_g – Carbohydrates in grams.

fiber_g – Fiber in grams.​

Example (structure only):

json
{
  "burger": {
    "calories": 258,
    "protein_g": 17,
    "fat_g": 2,
    "carbs_g": 30,
    "fiber_g": 1
  }
}
🧪 Data Preprocessing & Augmentation
Resize: All images → 224 × 224 × 3.

Normalize: Scale pixel values from to for stable training.​

Split:

Train: 70%

Validation: 15%

Test: 15%​

Grouping: Classes are grouped into Group_1 … Group_11 for modular training and evaluation.

Augmentation techniques (applied on training images only):

Rotation (small random angles)

Horizontal flip

Zoom (in/out)

Shear transforms

These operations increase data diversity and reduce overfitting.​

🧠 Model Architectures
Model	Depth	Key Idea	Usage in Project
VGG‑16	16 layers	Stacked 3×3 convolutions	High accuracy, fine‑tuned on food data
ResNet‑50	50 layers	Residual (skip) connections	Handles complex, deep feature learning
Custom CNN	4 conv + FC	Lightweight, task‑specific network	Fastest inference, suitable for edge
VGG‑16 (Defined)
13 convolutional layers + 3 fully‑connected layers.

Initialized with ImageNet weights and adapted to 34 classes by replacing the final classification layer.

Some early layers may be frozen; higher layers are fine‑tuned on the food dataset.​

ResNet‑50 (Defined)
50‑layer residual network with many residual blocks.

Each residual block adds its input to the output of several convolutional layers (skip connection).

Helps train deep networks without vanishing gradients; adapted to output 34 classes.​

Custom CNN (Defined)
4 convolutional layers with ReLU activation and MaxPooling.

One or more dense (fully‑connected) layers ending in a softmax layer with 34 outputs.

Designed to be smaller and faster than VGG‑16/ResNet‑50 while maintaining reasonable accuracy.​

📊 Training Setup & Metrics
Loss function: Categorical cross‑entropy (multi‑class).

Optimizers: Adam or SGD with momentum.

Hyperparameters:

Learning rate: typically 1e‑3 to 1e‑4 for fine‑tuning.

Batch size: 16–64 (depends on GPU).

Epochs: trained until validation performance stabilizes.

Metrics stored in model_performance.json:

Accuracy (overall and per group)

Precision, Recall, F1‑score per class

TP, FP, FN, TN counts per class​

Example snippet:

json
{
  "Model_Name": "VGG-16",
  "Model_File": "vgg16_group_1.h5",
  "Group_Name": "Group_1",
  "Test_Samples": 180,
  "Test_Accuracy": 88.75,
  "Classes": [
    {
      "Class_Name": "biryani",
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
Confusion matrices are generated to visualize per‑class performance.​

🧮 Nutritional Analysis Pipeline
Image input: User uploads a food image.

Preprocessing: Resize + normalize.

Prediction:

Choose a trained model (e.g., best performing VGG‑16 or ResNet‑50).

Model outputs predicted class and confidence.

Nutrition lookup:

Using predicted class as key, fetch values from food_nutrition.json.

JSON response:

json
{
  "Predicted_Class": "burger",
  "Confidence": 0.94,
  "Calories": 258,
  "Protein_g": 17,
  "Fat_g": 2,
  "Carbs_g": 30,
  "Fiber_g": 1
}
This is directly useful for calorie tracking, diet logs, and recommendation systems.​

🛠️ Tech Stack & Library Explanations
Python: Core language for data handling, modeling, and scripting.

TensorFlow / Keras (if used):

TensorFlow: Framework for building and training deep learning models with GPU support.

Keras: High‑level API on top of TensorFlow, providing layers like Conv2D, MaxPooling2D, Dense, and utilities for data generators.​

PyTorch (if used instead):

Deep learning framework with dynamic computation graphs, commonly used for research and flexible model design.​

NumPy: Efficient numerical operations on arrays (used for pre/post‑processing, metrics).

Pandas: Tabular data manipulation (analysis of results, nutritional tables, CSV/JSON handling).

Matplotlib / Seaborn: Plotting accuracy curves, loss curves, confusion matrices, and nutrition distributions.​

OpenCV or Pillow (PIL): Image loading, resizing, and basic image manipulations.

Flask / FastAPI (optional for deployment):

Flask: Lightweight Python web framework for simple REST APIs.

FastAPI: High‑performance web framework with automatic docs, good for ML model serving.

📁 Project Structure
text
.
├── data/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── food_nutrition.json
├── models/
│   ├── vgg16_group_*.h5
│   ├── resnet50_group_*.h5
│   └── custom_cnn_group_*.h5
├── results/
│   └── model_performance.json
├── notebooks/
│   └── exploration_and_training.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_vgg16.py
│   ├── train_resnet50.py
│   ├── train_custom_cnn.py
│   ├── evaluate_models.py
│   └── inference_api.py
└── README.md
▶️ How to Run
1. Clone
bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
2. Install Dependencies
bash
pip install -r requirements.txt
3. Prepare Data
Place images into data/train, data/val, data/test, each class in its own folder.

Ensure food_nutrition.json is inside data/.

4. Train Models
bash
python src/train_vgg16.py
python src/train_resnet50.py
python src/train_custom_cnn.py
5. Evaluate
bash
python src/evaluate_models.py
6. Inference on One Image
bash
python src/inference_api.py --image_path sample.jpg
🔮 Future Work
Real‑time mobile camera integration.

Portion size estimation (volume/area based).

Grad‑CAM heatmaps to explain model focus regions.

Cloud deployment (AWS/GCP/Render) with REST API.

Integration with fitness trackers and health apps.​

📚 References
Simonyan, K., Zisserman, A. (2014). Very Deep Convolutional Networks for Large‑Scale Image Recognition.​

He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition.​

Food‑101 Dataset (for inspiration in design): https://www.vision.ee.ethz.ch/datasets_extra/food-101/​

Works on food recognition and nutrition estimation using CNNs.​

👩‍💻 Author
Name: Nagalakshmi Ragolu
Education: B.Tech – Information Technology
Email: nagalakshmiragolu@gmail.com
LinkedIn: https://www.linkedin.com/in/ragolu-nagalakshmi-71587a22
