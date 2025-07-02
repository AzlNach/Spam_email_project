# Email Spam Classification Dashboard

## 📧 Project Overview
A comprehensive machine learning dashboard for email spam classification using multiple algorithms and interactive visualizations. This project implements both supervised learning (Logistic Regression) and unsupervised learning (K-Means Clustering) approaches to classify emails as spam or non-spam with high accuracy.

## ✨ Key Features

### 🎯 Core Functionality
- **Real-time Email Classification**: Interactive prediction system for new email data
- **Multi-Model Analysis**: Comparison between Logistic Regression and K-Means clustering
- **Advanced Data Preprocessing**: Outlier handling, noise reduction, and feature engineering
- **Interactive Dashboard**: User-friendly Streamlit interface with multiple visualization options

### 📊 Visualization Capabilities
- **Model Performance Metrics**: ROC curves, Precision-Recall curves, Confusion matrices
- **Clustering Analysis**: Elbow method, Silhouette analysis, PCA-based cluster visualization
- **Data Exploration**: Correlation heatmaps, statistical summaries, and distribution analysis

### 🔧 Technical Features
- **Automated Hyperparameter Tuning**: GridSearchCV optimization for Logistic Regression
- **Feature Engineering**: Custom cluster-based feature creation and probability assignment
- **Data Quality Enhancement**: Moving average smoothing, duplicate removal, correlation-based feature selection

## 🛠️ Technical Stack

### Libraries & Frameworks
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly Express, Plotly Graph Objects
- **Dimensionality Reduction**: PCA (Principal Component Analysis)

### Machine Learning Models
1. **Logistic Regression**
   - GridSearchCV hyperparameter optimization
   - Multiple penalty types (L1, L2, ElasticNet)
   - Various solvers (SAGA, LibLinear, LBFGS, etc.)

2. **K-Means Clustering**
   - Custom feature-based clustering approach
   - Optimal cluster determination using Elbow method and Silhouette analysis
   - Spam probability assignment per cluster

## 🔄 System Architecture & Data Flow

### 1. Data Preprocessing Pipeline
```
Raw Dataset → Outlier Detection & Handling → Duplicate Removal → 
Noise Reduction (Moving Average) → Correlation Analysis → 
Feature Selection → Missing Value Imputation → Normalization
```

### 2. Model Training Workflow
```
Preprocessed Data → Train/Test Split → 
├── Logistic Regression Path:
│   └── GridSearchCV → Best Model Selection → Performance Evaluation
└── K-Means Clustering Path:
    └── Feature Engineering → Cluster Assignment → Validation
```

### 3. Dashboard System Flow
```
User Interface → Navigation Selection → 
├── Data Description: Dataset overview and statistics
├── Model Evaluation: Performance metrics and prediction interface
├── Visualization: Interactive charts and analysis
└── Conclusion: Summary and insights
```

## 📈 Model Performance

### Logistic Regression Results
- **Accuracy**: 99.5%
- **ROC AUC Score**: 0.995
- **Precision**: 99.6%
- **Recall**: 99.6%
- **F1-Score**: 99.6%

### K-Means Clustering Results
- **Optimal Clusters**: 2
- **Silhouette Score**: 0.5667
- **Cluster Separation**: Clear distinction between spam and non-spam emails

## 🗂️ Dataset Features

### Word Frequency Features
- 48 features tracking frequency of specific words (e.g., "free", "business", "money", "credit")
- Values represent percentage of word occurrence in emails

### Character Frequency Features
- 6 features for special characters (!, $, #, ;, (, [)
- Important indicators for spam detection

### Capital Letter Features
- Average, longest, and total capital letter run lengths
- Spam emails often use excessive capitalization for emphasis

### Target Variable
- **Class**: Binary classification (1 = Spam, 0 = Non-Spam)

## 🚀 Getting Started

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

### Running the Application
```bash
streamlit run streamlit_app.py
```

### Development Environment
The project includes a `.devcontainer` configuration for consistent development setup:
- Python 3.11
- Pre-configured extensions (Python, Pylance)
- Automatic dependency installation
- Port forwarding for Streamlit (8501)

## 💡 Key Innovations

1. **Hybrid Clustering Approach**: Custom feature-based cluster assignment with probability weighting
2. **Advanced Preprocessing**: Multi-stage data cleaning with outlier handling and noise reduction
3. **Interactive Model Comparison**: Side-by-side analysis of different ML approaches
4. **Real-time Prediction**: User input interface for live spam classification

## 👥 Development Team
- **Abim Bimasena A.R.P** - S1 Sistem Informasi
- **Azel Pandya Maheswara N.A** - S1 Sistem Informasi  
- **Danendra Pandya Maheswara** - S1 Sistem Informasi
- **Sahal Fajri** - S1 Sistem Informasi

## 📝 Usage Instructions

1. **Data Exploration**: Navigate to "Deskripsi Data" to understand dataset characteristics
2. **Model Testing**: Use "Evaluasi Model" to input custom email features and get predictions
3. **Performance Analysis**: Check "Visualisasi" for detailed model performance metrics
4. **Results Review**: Read "Kesimpulan" for comprehensive analysis summary

## 🔮 Future Enhancements
- Integration with real email APIs
- Additional ML algorithms (Random Forest, SVM, Neural Networks)
- Real-time email processing capabilities
- Enhanced feature engineering techniques
- Model deployment for production use