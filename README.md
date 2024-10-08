# Dissertation-Dry-AMD-Project


# LEEDS BECKETT UNIVERSITY
**SCHOOL OF BUILT ENVIRONMENT, ENGINEERING AND COMPUTING**


## MONDry-FR: A Hybrid Model for the Prediction of Dry Age-related Macular Degeneration in MONAI using Fundus Retinography 

Submitted to Leeds Beckett University in partial fulfilment of the requirements for the degree of MSc Advance Computer Science

By ***Sameera Mushtaq Student ID: 77385572***

Supervised by ***[Kiran Vodherhobli](https://www.leedsbeckett.ac.uk/staff/kiran-voderhobli/)***

September, 2024


## Project Overview

This project involves:
- Exploring MONAI
- Robust hybrid model development for Dry AMD


## Getting Started

### Dependencies

Ensure you have the following dependencies installed:

- **Access to Google Colab**
-**Visual Studio Code (IDE)**
- **Docker**
-**Python**
-**GitHub**
## Setup Environment
### Backend Development:
Environment:
Google Colab: Used for training the deep learning models and preprocessing the dataset. Colab provides GPU support, which is crucial for handling computationally expensive deep learning tasks.
Technology Stack:
Python: Core programming language for the project.
PyTorch: Deep learning framework used for model training and evaluation.
MONAI: Medical Open Network for Artificial Intelligence, a specialized framework for developing deep learning models in medical imaging.
Torchvision: Used for pre-trained models (ResNet50, VGG16, etc.) and image transformations.
scikit-learn: For traditional machine learning models (KNN, Random Forest) and evaluation metrics (precision, recall, F1-score).
Streamlit: Used in the later stages for model deployment.
### Frontend Development:
Environment:

Visual Studio Code: For front-end code development.
Docker: For containerizing the frontend, ensuring that the web app can be deployed consistently across different environments.
Technology Stack:

Streamlit: Framework for creating the interactive web interface for Dry AMD detection.
Docker: Used to package the frontend application into a container, ensuring that all dependencies are properly managed and the application is easy to deploy across any system.


### Steps to run the Backend 
1. Click on the link provided to access drive 
   ```
   https://drive.google.com/drive/folders/1hY3hD1g8kbLUL_OheHuh0QsCYX4Lsq4O?usp=sharing 

   ```
2. Create a "Dry AMD" Folder in your " MY DRIVE".
3. Move "Fundus Images" folder and "MONDry-FR Model- Backend file" to your folder. 
4. Create a New folder named as " Model metrics". 
5. Right- Click on the file "MONDry-FR Model- Backend " and choose Open-With - Google Colaboratary
6. Once the file is opened, Click on the "Connect T4" from right top corner.
7. Click on "RUN ALL".
8. At step 3 " It will ask you to connect to your drive. Please provide access of your drvie. 
NOTE: Names are Case Sensitive 
### Steps to view the Backend File
1. Click on the link provided to access drive 
   ```
   https://drive.google.com/drive/folders/1hY3hD1g8kbLUL_OheHuh0QsCYX4Lsq4O?usp=sharing 

   ```
2. Right- Click on the file "MONDry-FR Model- Backend " and choose Open-With - Google Colaboratary

OR
2. Go to Google Colab site and upload MONDry-FR Model- Backend.ipynb file to view it.

### Steps to Run the Application - Front End


1. Clone the repository by running this in the terminal

   ```bash
      git clone https://github.com/2Sam392/Dissertation-Dry-AMD-Project.git 
   ```
2. Run the Docker first 
3. Open your terminal and run the following command to install the required dependencies:

   ```bash
      code . 
   ```

4. Open Terminal and type the following command

   ```
   docker-compose up 
   ```

5. Click on the URL provided to open the link on your borwser
6. Upload a Healthy and Normal Retina images from the Test folder in the cloned repository
7. The application will generate the results.

## To Exit Front-end
1. Go to the Terminal and Press "CTRL+C" several times until it stops. 
2. Type " docker-compose down"
2. Quit the Docker.

