# Dogs Identifier

## Table of Contents
1. [Introduction](#introduction)
2. [Technologies](#technologies)
3. [Microservices Description](#microservices-description)
4. [Setup Guide](#setup-guide)
5. [Visual Presentation](#visual-presentation)

## Introduction
This project is designed for learning how to use Docker.

The result of this project is a set of microservices communicating with each other, consisting of a machine learning module, database, and a visual interface in the form of a web app.

Project features include:
- Detecting dog breeds using photos.
- Adding dogs with their picture, breed, and curiosity to the database.
- Learning about dogs with a list of their breeds, curiosities, and photos.

## Technologies

This project was made with:

- ![Docker](images/docker.png) **Docker**: 26.1.1
- ![MySQL](images/mysql.png) **MySQL**: 8.3.0
- ![Python](images/python.png) **Python**: 3.9
- ![Flask](images/flask.png) **Flask**: 3.0.3
- ![Streamlit](images/streamlit.png) **Streamlit**: 1.33.0

## Microservices Description

### Database Server
This service is responsible for providing the database. It is based on MySQL server.

### UI
This service is responsible for the visual interface of the application. It allows users to use the functionalities of the app. It is based on Streamlit.

### Flask
This service is responsible for communicating between the database and UI. It is based on Flask.

## Setup Guide

### Prerequisites

To install this software and be able to run it, you need to install Docker on your machine. If you don't have Docker, you can download it from the [official distribution](https://www.docker.com/).

### Installing the Application 

After you clone the repository, you need to run the terminal and go to the directory `\Wetwpy\Dockerized`, which contains all the files.

Next, you need to set up the application by building the container.

```bash
docker compose up -d
```


