# BTT AI WiDS UCLA Team 10

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Anne Do | @doanneda | Built GNN model, performed data augmentation |
| Abhi Sathvika Goriparthy| @abhi-sathvika | Built and optimized GNN Model, performed hyperparameter tuning to reduce overfitting |
| Haley Lepe | @haleylepe | Implemented EDA and Random Forest Classifier |
| Ashley | @ashley | Visualized dataset distributions, handled missing data |

---

## **🎯 Project Highlights**
* Built a Graph Neural Network (GNN) model using PyTorch Geometric to predict ADHD diagnosis and sex based on functional connectome matrices.
* Achieved an accuracy of 0.75054 and ranked 129 out of 529 on the final WiDS Datathon Kaggle Leaderboard.
* Implemented cosine similarity-based edge construction to form a single large graph representing relationships between subjects.
* Used Dropout and L2 Regularization to prevent overfitting, improving generalization performance on the test set.
* Optimized data preprocessing and graph construction to handle large-scale data efficiently within compute constraints.


🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

**Clone the Repository**  
- First, clone this GitHub repository to your local machine:
```
git clone https://github.com/your-username/widsdatathon25.git
cd widsdatathon25
```

**Download the dataset from Kaggle:**
- Place the downloaded CSV files into the data/ directory

**Run the Jupyter Notebook on Google Colab**
- Make sure to download all the packages listed in the imports

---

## **🏗️ Project Overview**

The WiDS Datathon 2025 is a worldwide competition dedicated to giving individuals an opportunity to explore data with the intent to bring awareness to ADHD in women across the world. With Break Through Tech’s mission being to empower underrepresented students in AI, such as women and nonbinary individuals, it is important to use the knowledge BTT has supplied to its students to bring awareness to similar communities, and use AI for the common good.

The objective of the challenge is to identify key factors associated with undiagnosed ADHD in women using data-driven insights, such as socio-demographic, emotions, and parenting information. By exploring trends and disparities, participants aim to develop models that can improve early detection and understanding of ADHD in underrepresented populations. 

ADHD in women is frequently misdiagnosed or overlooked, leading to challenges in education, mental health, and daily life. Through this competition, participants have the opportunity to create predictive models that could inform healthcare professionals, shape public policy, and ultimately lead to improved screening and diagnosis methods. The work done in this challenge has the potential to bridge gaps in healthcare equity and bring meaningful change to communities worldwide.

---

## **📊 Data Exploration**

For our project, we utilized the available training data from the WiDS 2025 competition, which was sourced from Cornell and UC Santa Barbara. We utilized a variety of includes which include one-hot encoding for our categorical variables, handled missing data, merged dataframes together and other techniques in our GNN model. We made visualizations to gain insights into our data, such as bar plot and box plots. 

<img src="https://github.com/user-attachments/assets/255347eb-c17a-42f8-9c08-d393575e73c1" width="500" height="auto">
<img src="https://github.com/user-attachments/assets/7b897021-f77d-4c63-8c2d-beae68613c30" width="500" height="auto">
<img src="https://github.com/user-attachments/assets/327f9849-69da-4e4e-be66-9dc7f382c643" width="500" height="auto">

---

## **🧠 Model Development**  

### **Model(s) Used**  
We implemented a **Graph Neural Network (GNN)** using **Graph Convolutional Networks (GCN)** via **PyTorch Geometric**. The model leverages **cosine similarity-based edge construction** to connect subjects based on functional brain activity.  

### **Feature Selection & Hyperparameter Tuning**  
- **Feature Selection**: Functional connectome matrices were used as node features, while edges were constructed based on cosine similarity.  
- **Edge Threshold Optimization**: We experimented with different similarity thresholds (e.g., 0.2, 0.25) to find an optimal balance between graph sparsity and connectivity.  
- **Hyperparameter Tuning**:  
  - Learning rate: **0.01 → 0.005** (reduced to prevent overfitting)  
  - Hidden dimensions: **64 → 32** (simplified model for better generalization)  
  - Dropout: **0.3** (to mitigate overfitting)  
  - Weight decay: **1e-4** (L2 regularization to improve generalization)  

### **Training Setup**  
- Train-Validation Split:  
  - 80% of subjects used for training  
  - 20% reserved for validation  
-Evaluation Metric: Weighted F1-score (due to class imbalance)  
- Baseline Performance: We initially experimented with an **XGBoost baseline** using metadata features, but the **GNN outperformed it significantly** due to its ability to model relationships between subjects.  

---

## **📈 Results & Key Findings**  

### **Performance Metrics**  
- **Final Kaggle Rank**: 129/529 
- **Accuracy**: 0.75054
- **Loss & Convergence**: Training loss stabilized around 0.875

### **Model Performance**  
- **Overall**: The GNN performed well in capturing functional brain connectivity to predict ADHD and sex.  
- **Across ADHD vs. Non-ADHD**: The model performed better for males than females, indicating potential biases in the dataset.  
- **Across Different Features**: The connectivity-based edges significantly improved performance compared to a fully connected graph.  

---

## **🖼️ Impact Narrative**  

### **1️⃣ What brain activity patterns are associated with ADHD?**  
Our findings suggest that certain connectivity patterns in the brain are strongly linked to ADHD diagnosis.
- The difference between males and females was evident in the strength of connectivity in certain brain regions, suggesting possible neurobiological distinctions.  

### **2️⃣ How could our work contribute to ADHD research or clinical care?**  
- **Early Identification**: The model could help in identifying individuals at risk for ADHD based on brain connectivity patterns.  
- **Personalized Treatment**: By understanding sex-specific differences, clinicians could tailor ADHD interventions for males and females.  
- **Scalability**: The graph-based approach allows for scalability across different datasets, potentially aiding broader neurodevelopmental research.  

---

## **🚀 Next Steps & Future Improvements**  

**Limitations of Our Model**  
- Potential Overfitting: Although we applied dropout and L2 regularization, the model still exhibited signs of overfitting, indicating that further hyperparameter tuning or a larger dataset may be required.  
- Data Imbalance: The dataset may have unequal representation of ADHD cases across sexes, which could introduce bias in predictions.  
- Graph Construction Approach: Using cosine similarity for edge creation was effective, but alternative methods (K-nearest neighbors (KNN), correlation-based graphs) could further refine connectivity representations.  

### **Future Improvements with More Time/Resources**  
- **Hyperparameter Optimization**: Conduct extensive tuning using Bayesian Optimization or Grid Search to refine learning rate, dropout rate, and edge thresholds.  
- **Alternative GNN Architectures**: Explore Graph Attention Networks (GAT) or GraphSAGE to improve information propagation and capture non-local dependencies.  
- **Fairness Adjustments**: Apply data augmentation or class balancing to reduce bias in ADHD predictions across genders.  
- **Multi-Graph Representations**: Instead of a single subject-to-subject graph, experiment with region-wise connectivity graphs within each brain to capture finer relationships.  

---

## **📄 References & Additional Resources**

- [WiDS Datathon 2025 Datasets](https://www.kaggle.com/competitions/widsdatathon2025/data)  
- [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)  
- [The Role of AI in Advancing Women's Brain Health Research](https://youtu.be/7X4M8dYkrvw) 

---
