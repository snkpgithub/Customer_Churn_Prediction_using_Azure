# Customer Churn Prediction using Azure Machine Learning

This project demonstrates a data science workflow for predicting customer churn using Azure Machine Learning. It showcases skills in data preprocessing, machine learning, and cloud computing with Azure.

## Project Overview

Customer churn prediction is a critical task for businesses to identify customers who are likely to stop using their services. This project uses Azure Machine Learning to automate the process of finding the best model for churn prediction.

### Key Features

- Data preprocessing and exploration
- Automated Machine Learning (AutoML) with Azure ML
- Model registration and deployment
- Integration with Azure cloud services

## Technologies Used

- Python
- pandas
- scikit-learn
- Azure Machine Learning
- Azure AutoML

## Getting Started

### Prerequisites

- Azure account with an active subscription
- Azure Machine Learning workspace
- Python 3.7+
- Required Python packages: pandas, numpy, scikit-learn, azureml-core, azureml-train-automl-runtime

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/customer-churn-prediction.git
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Azure Machine Learning workspace and update the `config.json` file with your workspace details.

### Usage

1. Prepare your customer churn dataset and update the path in the script.
2. Run the main script:
   ```
   python customer_churn_prediction.py
   ```

## Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── customer_churn_data.csv
│
├── src/
│   ├── customer_churn_prediction.py
│   └── utils.py
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── config.json
├── requirements.txt
└── README.md
```

## Results

After running the AutoML experiment, the best model is selected based on the AUC (Area Under the Curve) metric. The model is then registered in Azure ML for future use or deployment.

## Future Improvements

- Implement a web interface for real-time predictions
- Enhance the model with feature engineering
- Explore deep learning models for churn prediction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Azure Machine Learning team for their comprehensive documentation
- The open-source community for providing valuable resources on churn prediction

