# FastAPI Book Recommendation System

This repository contains a FastAPI application for recommending books to users based on their ratings. The application loads a pre-trained machine learning model to predict book ratings and provides book recommendations for a given user.

## Table of Contents
- [Features](#features)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running Locally](#running-locally)
  - [Running with Docker](#running-with-docker)
  - [Deployment to Google Cloud Run](#deployment-to-google-cloud-run)
- [API Endpoints](#api-endpoints)

## Features
- Load and preprocess data from Google Firestore.
- Use a pre-trained TensorFlow model to predict book ratings.
- Provide book recommendations for a given user.

## Setup

### Prerequisites
- Python 3.9+
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- Docker

### Environment Variables
Create a `.env` file in the root directory and add your Google Cloud credentials:


### Installation
1. Clone the repository:
   ```sh
    git clone https://github.com/your-repo/book-recommendation-system.git
    cd book-recommendation-system
    pip install -r requirements.txt

### Usage
1.  Running Locally
    Set up your Google Cloud credentials:
    ```sh
    export GOOGLE_APPLICATION_CREDENTIALS="serviceAccountFireStore.json"

### Run the FastAPI application:`
```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

**Query Parameter**:
- `user_id` (integer, required): The ID of the user for whom you want to fetch book recommendations.

**Example Request**:
```http
GET https://api.example.com/v1/predict?user_id=123