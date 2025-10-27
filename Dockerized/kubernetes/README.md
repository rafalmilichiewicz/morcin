# Prerequisites

-   Docker
-   Minikube
-   Minikube addons
    -   ingress: `minikube addons enable ingress`
    -   metrics-server: `minikube addons enable metrics-server`
    -   registry: `minikube addons enable registry`

# Load image

minikube image load your-app-image:latest

# Commands

```sh
# Namespace
kubectl create namespace dockerized -o yaml --dry-run=client -o yaml > ./kubernetes/00_namespace.yaml


# Resource quota
kubectl create quota dockerized-quota \
  -n dockerized \
  --hard=pods=6,requests.cpu=2,requests.memory=1Gi,limits.cpu=4,limits.memory=2Gi \
  --dry-run=client -o yaml > ./kubernetes/01-resourcequota.yaml

# Secrets
kubectl create secret generic mysql-root-pass \
  -n dockerized \
  --from-literal=MYSQL_ROOT_PASSWORD='password' \
  --dry-run=client -o yaml > ./kubernetes/02-mysql-secret.yaml

# Config map
kubectl create configmap app-config \
  -n dockerized \
  --from-literal=DB_HOST='mysql' \
  --from-literal=DB_USER='root' \
  --from-literal=DB_NAME='dogs_db' \
  --dry-run=client -o yaml > ./kubernetes/03-app-configmap.yaml



# Database


## Stateful Set
# No command to create - copied and edited from Kubernetes docs

## Service
kubectl create service clusterip mysql \
  --tcp=3306:3306 \
  -n dockerized \
  --dry-run=client -o yaml > ./kubernetes/05-mysql-headless-svc.yaml


# PHP My Admin
## Deployment
kubectl create deployment phpmyadmin \
  --image=phpmyadmin/phpmyadmin:latest \
  -n dockerized \
  --dry-run=client -o yaml > ./kubernetes/06-phpmyadmin-deployment.yaml

# Edited to add resource quotas and env variables

## Service
kubectl create service loadbalancer phpmyadmin \
  --tcp=80:8080 \
  -n dockerized \
  --dry-run=client -o yaml > ./kubernetes/07-phpmyadmin-svc.yaml


# App

## Docker image
docker build -t python-app:latest -f dockerfile .
docker save --output python-app.img python-app:latest
minikube image load python-app.img

## Deployment
kubectl create deployment app \
  --image=python-app:latest \
  -n dockerized \
  --dry-run=client -o yaml > ./kubernetes/08-app-deployment.yaml

## Added imagePullPolicy: Never to prevent problem with k8s trying to download unpublished image

## Service
kubectl create service loadbalancer app \
  --tcp=8501:8501 \
  -n dockerized \
  --dry-run=client -o yaml > ./kubernetes/09-app-svc.yaml

# Ingress
kubectl create ingress dockerized-ingress \
  -n dockerized \
  --rule="python.app/*=app:8501" \
  --rule="pma.python.app/*=phpmyadmin:80" \
  --dry-run=client -o yaml > 10-ingress.yaml

# Added ingress.kubernetes.io/rewrite-target: /
```

# Running

```sh
kubectl apply -f 00_namespace.yaml
kubectl apply -f 01-resourcequota.yaml
kubectl apply -f 02-mysql-secret.yaml
kubectl apply -f 03-app-configmap.yaml
kubectl apply -f 04-mysql-stateful-set.yaml
kubectl apply -f 05-mysql-headless-svc.yaml
kubectl apply -f 06-phpmyadmin-deployment.yaml
kubectl apply -f 07-phpmyadmin-svc.yaml
kubectl apply -f 08-app-deployment.yaml
kubectl apply -f 09-app-svc.yaml
```
