# kuzushiji-49-inference-server

This project aims to develop and deploy a machine learning model capable of recognizing Kuzushiji, traditional Japanese cursive script, from images. We process and classify images of handwritten Kuzushiji characters into 49 different Hiragana classes. The project encapsulates the entire workflow from data preprocessing, model training and evaluation, to the final deployment of a FastAPI-based inference server, making it possible to predict Kuzushiji characters from new images through a simple API call.

## Kuzushiji-49 Dataset

The Kuzushiji recognition task is based on the KMNIST dataset, which is a collection of handwritten Hiragana characters. The dataset is designed to serve as a drop-in replacement for the MNIST dataset, offering a more challenging task and promoting the preservation and study of this important aspect of Japanese cultural heritage. For more information about the dataset and to download it, please visit [this link](https://github.com/rois-codh/kmnist?tab=readme-ov-file#kuzushiji-49).

## Project Highlights
- **Data Preprocessing**: Implements comprehensive image preprocessing steps including resizing and normalization to prepare the dataset for efficient model training.
- **Model Training**: Designs and trains a few models using PyTorch, fine-tuned to recognize Kuzushiji characters with high accuracy.
- **Model Evaluation**: Employs various metrics to evaluate the model's performance, ensuring it meets the expected accuracy standards.
- **API Server Deployment**: Deploys the trained model using FastAPI, creating a robust and scalable inference server ready for real-world applications.
- **Ease of Use**: Offers an easy-to-use API for making predictions, streamlining the process of Kuzushiji character recognition in digital images.

## Technologies Used

### Device
- **Apple M1 Pro**
- **macOS**: 13.4.1

### Infrastructure
- **Docker**: 20.10.17, build 100c701

### Machine Learning
- **Python**: 3.10.10
- **PyTorch**: 2.2.1

### API Development
- **FastAPI**: 0.110.0
- **Uvicorn**: 0.29.0

## Project Structures
- `app/`: Contains the FastAPI application for the inference server.
  - `inference/`: Inference-related modules including model definitions (`architectures`), prediction handlers (`handlers`), and utility functions (`utils`).
- `data/`: Data directory for raw datasets and uploaded images.
- `infra/`: Docker and Docker Compose files for setting up the development environments.
- `log/`: Logs from training and inference processes.
- `model/`: Trained model artifacts.
- `notebook/`: Jupyter notebooks for data exploration and analysis.
- `training/`: Scripts and modules for training the models.
  - `download/`: Scripts for downloading and preparing the dataset.
  - `model/`: 
    - `architectures/`: Definitions of neural network architectures used in the project.
    - `model_builders/`: Scripts and modules responsible for assembling the model architecture, loading pre-trained weights if available, and preparing the model for training. This includes setting up the optimizer, loss functions, and any specific model configurations.
  - `utils/`: Utility functions and classes for training, including data loaders, custom loss functions, and performance metrics.
  - `evaluator.py`: Script for evaluating trained models against a validation set.
  - `trainer.py`: Main training script that orchestrates the training process, including model training, validation, and logging.

## Environment Setup

To set up the environment for this project, Docker and Docker Compose are required. This ensures that all dependencies and services are correctly installed and configured.

Follow these steps to set up your environment:

1. Install Docker and Docker Compose on your system if you haven't done so already. Please refer to the [official Docker documentation](https://docs.docker.com/get-docker/) for installation instructions.

2. Clone this repository to your local machine. You can do this by running the following command in your terminal:
    ```sh
    git clone https://github.com/koroku1023/kuzushiji-49-inference-server.git
3. Change directory:
    ```sh
    cd kuzushiji-49-inference-server
    ```
4. Build and start the Docker containers using Docker Compose. Run the following command:
    ```sh
    docker-compose -f infra/docker-compose.yml up --build -d
    ```
5. To confirm that the containers are running successfully, use:
    ```sh
    docker-compose ps
    ```

## Model Training Execution Method

### Supported Models

- **Simple CNN**
- **DenseNet121**

### Executing the Training Process

To train a model, follow these steps:

1. Ensure that the Docker environment is set up and running as described in the [Environment Setup](#environment-setup) section.

2. To start the training process for a specific model, use the following command in the terminal:
    ```sh
    docker-compose -f infra/docker-compose.yml exec jupyter python training/model/model_builders/{model_name}.py
    ```
    Replace {model_name} with the name of the model you wish to train. Supported model names are `cnn`(Simple_CNN), `densenet`(DenseNet121).

### Output Locations
After the training process completes, the model artifacts and logs are saved to the following locations within the Docker container:

- Model Artifacts: The trained model files are saved in the model/ directory. For the CNN model, the output file will be named simple_cnn.pth.
- Training Logs: The training logs, which include details about the training process such as loss and accuracy metrics, are saved in the log/training/ directory. The log files are named according to the model and the timestamp of the training session, e.g., 20240320_141416_simple_cnn.log.

## Model Accuracy Table

To evaluate the performance of the trained models on the test dataset, we use several metrics including accuracy, precision, recall, and F1 score.

| Model Name | Accuracy | Precision | Recall | F1 Score |
|------------|----------|-----------|--------|----------|
| CNN        |0.7691    | 0.7667    | 0.7421 | 0.7480   |
| DenseNet121| 0.9237   | 0.9134    | 0.9125 | 0.9123   |

## Inference Using the Inference Server

To perform inference using the trained model through the FastAPI inference server, you can use the `curl` command from the command line. This section guides you on how to send a test image to the server and receive the inference results.

### Prerequisites

Ensure the FastAPI server is running. If it's not running, please refer to the [Environment Setup](#environment-setup-method) section to start the server.

### Available Models and Test Data

- **Models**:
  - `cnn`
  - `densenet`
- **Test Data**:
  - Use `test_image_0.npz` for a single image test.
  - Use `test_images_10.npz` for testing with a batch of 10 images.
  - Use `test_images_100.npz` for testing with a batch of 100 images.
  - `data/raw/k49-test-imgs.npz` can be used to test the server's handling of larger data sets.


### Sending Synchronous Inference Request

#### Parameters

- **model_name**: (Required, String) The name of the model to use for the inference. 
- **file_name**: (Required, String) The name of the file containing the images for inference.

#### Response

The server will return the inference results in a JSON format. 

- **results**: (Array) Contains the prediction results for each image in the uploaded file. In case of an error, a descriptive error message is returned as a string.

**example**
```
{
  "model": "cnn",
  "task_id": "YYYYMMDD_hhmmss_XXXXX",
  "results": [
    {
      "index": 0,
      "start_timestamp": "20240320_095303",
      "prediction": "と",
      "prediction_label": "19",
      "probability": "0.31642184"
    }
  ]
}
```

#### Command

```sh
curl -X 'POST' \
  'http://localhost:8000/predict/{model_name}' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'upload_file=@data/test/{file_name};type=application/x-npz'
```

**example**

```sh
curl -X 'POST' \
  'http://localhost:8000/predict/cnn' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'upload_file=@data/test/test_image_0.npz;type=application/x-npz'
```

### Sending Asynchronous Inference Request

Parameters and Response are the same as Synchronous Inference.

#### Command

```sh
curl -X 'POST' \
  'http://localhost:8000/async_predict/{model_name}' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'upload_file=@data/test/{file_name};type=application/x-npz'
```

**example**

```sh
curl -X 'POST' \
  'http://localhost:8000/async_predict/cnn' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'upload_file=@data/test/test_image_0.npz;type=application/x-npz'
```

### Sending Batch Inference Request

#### Parameters
model_name: (Required, String) The name of the model for inference.
file_name: (Required, String) The name of the file containing images for inference.
execute_at: (Required, datetime in ISO 8601 format) Schedule time for the batch inference job. Format: YYYY-MM-DDTHH:MM:SSZ. TimeZone is UTC.

#### Response
The server returns a confirmation message indicating the scheduled time for the batch inference task.

**example**
```
{
  "message": "Batch prediction task scheduled at 2024-03-22T15:51:00Z"
}
```

#### Command
```sh
curl -X 'POST' \
  'http://localhost:8000/schedule/{model_name}?execute_at=YYYY-MM-DDThh%3Amm%3AssZ' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'upload_file=@data/test/{file_name};type=application/x-npz'
```

**example**

```sh
curl -X 'POST' \
  'http://localhost:8000/schedule/cnn?execute_at=2024-03-22T15%3A51%3A00Z' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'upload_file=@data/test/test_image_0.npz;type=application/x-npz'
```

#### Output
After the scheduled batch inference task is executed, the inference results are saved in the data/output/batch/ directory. Each result file is named using the task_id generated during scheduling, ensuring a unique filename for each batch inference task.

## Cleanup

```
# Stopping Containers
docker-compose -f infra/docker-compose.yml stop inference-server jupyter

# Removing Containers
docker-compose -f infra/docker-compose.yml rm inference-server jupyter

```
