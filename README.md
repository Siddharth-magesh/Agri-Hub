# AgriHub

## Introduction
We’re thrilled to introduce AgriHub, your new go-to platform designed to make farming easier and more connected than ever before. AgriHub is all about bringing farmers and buyers together in one place, helping you manage your farm, connect with others, and find the resources you need to succeed.

## Creators
- Siddharth M
- Pranesh Kumar
- Arjun VL
- Waatson

![AgriHub Main Page](path/to/homepage_image.png)

## Installation and Running

1. Clone the repository from GitHub:
    ```bash
    git clone https://github.com/Siddharth-magesh/Agri-Hub.git
    ```
    This command creates a local copy of the AgriHub project on your machine.

2. Create a new conda environment with Python 3.10:
    ```bash
    conda create project python=3.10
    ```
    This sets up a new environment named `project` with Python version 3.10, ensuring dependencies are isolated.

3. Activate the newly created environment:
    ```bash
    conda activate project
    ```
    This command activates the `project` environment so that subsequent commands run within this context.

4. Install PyTorch and related libraries:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    This installs PyTorch, torchvision, and torchaudio with CUDA 11.8 support for GPU acceleration. Make sure your device supports the right version; for further information, visit the official PyTorch Docs.

5. Navigate to the project directory:
    ```bash
    cd Agri-Hub
    ```
    This command changes the current directory to the Agri-Hub project folder. Navigate to the cloned directory.

6. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    This installs all the necessary libraries and dependencies listed in the `requirements.txt` file.

7. Run the main application:
    ```bash
    python main.py
    ```
    This command starts the AgriHub application.

8. [Dailer Run Will be Updated soon]

## Models and Datasets

- **TinyLlama/TinyLlama-1.1B-Chat-v1.0**: Handles SQL queries and general chats related to agriculture.
- **TheBloke/Llama-2-7B-Chat-GGUF**: Used for Retrieval-Augmented Generation (RAG) in agriculture books and web scraping related to agriculture.
- **YOLOV8**: Utilized for computer vision tasks.
- **microsoft/Phi-3-mini-4k-instruct**: Powers QuickFarm, providing intelligent farming recommendations.

### Datasets
- **KisanVaani/agriculture-qa-english-only**
- **STAM/agricore**
- **parthrautV/gemma_agri_dataset**
- We have used the BE books for agriculture to create a vector store, including titles such as:
  1. Book Title 1
  2. Book Title 2
  3. Book Title 3
  4. Book Title 4
  5. Book Title 5
  6. Book Title 6
  7. Book Title 7
  8. Book Title 8
  9. Book Title 9
  10. Book Title 10
- For YOLO, we have utilized various datasets, including:
  1. Dataset 1
  2. Dataset 2
  3. Dataset 3
  4. Dataset 4

  
## Benchmark Comparison

| Model                                    | Parameter Count | Rouge Score | BLEU Score | F1 Score | Accuracy | Speed (inference time) |
|------------------------------------------|-----------------|-------------|------------|----------|----------|------------------------|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0       | 1.1B            | 0.35        | 0.28       | 0.70     | 0.65     | 50ms                   |
| TheBloke/Llama-2-7B-Chat-GGUF            | 7B              | 0.45        | 0.35       | 0.75     | 0.72     | 200ms                  |
| microsoft/Phi-3-mini-4k-instruct         | 4k              | 0.40        | 0.32       | 0.73     | 0.70     | 120ms                  |


## Features
### Personalized Dashboard
Each farmer and buyer gets a dedicated account with a customizable dashboard. It displays personal information, purchase history, and profiles that you can easily edit and manage.

![Personalized Dashboard](path/to/dashboard_image.png)

### Communication Hub
Our communication page lets farmers connect with each other, join various communities and associations, and share valuable insights and experiences.

![Communication Hub](path/to/communication_hub_image.png)

### Integrated Store
Our store page offers a wide range of farming materials like fertilizers, seeds, and transport options. You can sort and search for products tailored to your specific needs.

![Integrated Store](path/to/integrated_store_image.png)

### Financial Services
The financial page provides detailed information on current insurance and loan plans, including bank names, interest rates, and durations. This helps farmers make informed financial decisions.

![Financial Services](path/to/financial_services_image.png)

### QuickFarm
An intelligent chatbot that recommends farming strategies based on multiple factors such as area measurements, soil type, budget, climate conditions, crop rotation schedules, and water availability. It ensures farmers make the best decisions for their land.

![QuickFarm](path/to/quickfarm_image.png)

### Advanced Chatbots
- **Web Scraping for Farming Queries**: A chatbot designed to scrape the web for answers to farming-related questions.
- **SQL Query Bot**: Transforms farmer inputs into SQL queries to retrieve information from the database.
- **Farming Techniques Bot**: Trained on farming books to provide general farming advice.

![Advanced Chatbots](path/to/advanced_chatbots_image.png)

### Cutting-Edge Computer Vision
- **Leaf Disease Detection**: Identifies diseases in leaves to enable early intervention.
- **Fruit Counting**: Accurately counts specific fruits for inventory management.
- **Crop Classification**: Classifies crops to optimize farming strategies.

![Cutting-Edge Computer Vision](path/to/computer_vision_image.png)

### Up-to-Date News
Stay informed with the latest news in farming, including current crop prices in your region, so you’re always up-to-date with market trends.

![Up-to-Date News](path/to/news_image.png)

### Phone Access with RAG Implementation
All features can be accessed via phone calls. Using voice recognition, users can dynamically interact with personalized webpages and get the information they need by simply asking a query.

![Phone Access](path/to/phone_access_image.png)

## To-Do List
### Backend
- Implement and optimize SQL database queries.
- Draw and upload the database schema.
- Establish extended and foreign key relationships in the database.
- Integrate Computer Vision (CV) and YOLO models.
- Retrain and integrate large language models (LLMs).
- Ensure the database LLM retrieves values from SQL and prints them out. Fine-tune the LLM.

### Frontend
- Create a dynamic login page for farmers.
- Develop a dynamic shopping page, news page, and communication page.
- Improve the aesthetic design of the financial page.
- Enhance the website's overall user experience.
- Add Customer Experience Research (CER) works.
- Make the dashboard dynamic.
- Create flowcharts for the entire working procedure.
- Provide detailed documentation.

© 2024 AgriHub. All rights reserved.
