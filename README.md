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
![Main Page](images/appscreens/mainpage.png)
**Main Page**

![Detection dogs breed](images/appscreens/detection.png)
**Detection dogs breed**

![Discover dog](images/appscreens/discover.png)
**Discover dog**

![Edit dog info](images/appscreens/edit.png)
**Edit dog info**

## machine-learning-content

![Training History](Dockerized/app/model/plots/training_history.png)