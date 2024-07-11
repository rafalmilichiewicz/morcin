# Dogs Identifier

### Streamlit Content
1. [Introduction](#introduction)
2. [Technologies](#technologies)
3. [Setup Guide](#setup-guide)
4. [Visual Presentation](#visual-presentation)

### ML Content
5. [Introduction to Machine Learning Content](#machine-learning-content)
6. [ML Technologies](#machine-learning-technologies)
7. [ML metrics](#machine-learning-metrics)
8. [ML description](#machine-learning-description)

## Introduction
This project is designed for learning ML and streamlit.

The result of this project is a web app consisting of a machine learning module, database, and a visual interface.

Project features include:
- Detecting dog breeds using photos.
- Adding dogs with their picture, breed, and curiosity to the database.
- Learning about dogs with a list of their breeds, curiosities, and photos.

## Technologies

This project was made with:

- ![MySQL](images/mysql.png) **MySQL**: 8.3.0
- ![Python](images/python.png) **Python**: 3.9
- ![Streamlit](images/streamlit.png) **Streamlit**: 1.33.0

## Setup guide

### Prerequisites

To install this software and be able to run it, you need to install packages from requirements.txt and run mysql server

### Installing the Application 

After you clone the repository, you need to run the terminal and go to the directory `\Dockerized\app`, which contains all the files.

```bash
streamlit run app.py
```


## Visual Presentation

<div align="center">
  <img src="images/appscreens/mainpage.png" alt="Main Page" /><br />
  <strong>Main Page</strong>
</div>

<br />

<div align="center">
  <img src="images/appscreens/detection.png" alt="Detection dogs breed" /><br />
  <strong>Detection dogs breed</strong>
</div>

<br />

<div align="center">
  <img src="images/appscreens/discover.png" alt="Discover dog" /><br />
  <strong>Discover dog</strong>
</div>

<br />

<div align="center">
  <img src="images/appscreens/edit.png" alt="Edit dog info" /><br />
  <strong>Edit dog info</strong>
</div>

## Introduction to Machine Learning Content

Pamiętaj o opisaniu skąd jest pobierany zbiór danych i opisanie zbioru danych

Projekt został
Jeżeli chce się ponownie wytrenować model z większą ilością epok lub zmienić jego ustawienia należy zainstalować nvidia cuda 12.1
Name: torch
Version: 2.2.2
(myenv) C:\Users\Admin>conda list | grep torch
pytorch                   2.2.2           py3.9_cuda12.1_cudnn8_0    pytorch
pytorch-cuda              12.1                 hde6ce7c_5    pytorch
pytorch-mutex             1.0                        cuda    pytorch
torchaudio                2.2.2                    pypi_0    pypi
torchvision               0.17.2                   pypi_0    pypi

pozostałe pakiety są w requirements.txt w sekcji ML
Należy zainstalować 
Projekt został zaprojektowany przy użyciu Restnet50 wraz z optimizerem Adam.
Tutaj mamy pokazaną miarę top-3

Opisać te miary i jak się je liczy 

metryke ogólnie 
porównanie dla przykładowych obrazków - policzyć np dla 3 obrazków i dać wyniki.

Opisać fragmenty kodu.

<div align="center">
  <img src="Dockerized/app/model/plots/training_history.png" alt="Training History" /><br />
  <strong>Training History</strong>
</div>
