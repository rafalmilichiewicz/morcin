# Prerequisites

-   Docker
-   Minikube
-   Kubectl
-   Minikube addons
    -   ingress: `minikube addons enable ingress`
    -   metrics-server: `minikube addons enable metrics-server`
    -   registry: `minikube addons enable registry`

# Kubernetes manifests

To obtain manifests initially these commands were used, then they were either modified or in some cases created by hand - `kubectl` does not support many options from cli.

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
  --dry-run=client -o yaml > ./kubernetes/10-ingress.yaml

# Added ingress.kubernetes.io/rewrite-target: /
```

# Kubernetes overview

-   Namespace: `dockerized`
-   Resource quota:
    -   Limits:
        -   CPU: `4`
        -   Memory: `6Gi`
        -   Pods: `10`
    -   Requests:
        -   CPU: `2`
        -   Memory: `3Gi`
-   Secrets - Base64 encoded MySQL database password: `password`
-   ConfigMap - used for `app` and `phpmyadmin` configuration
-   Apps:
    -   MySQL 9.5 - Stateful Set with health checks
    -   PHP MyAdmin - 1 replica
    -   App - Python application with health checks from local docker image
    -   Websocket - Python asyncio websocket app
-   Services:
    -   MySQL - ClusterIP - restricted access
    -   PHP MyAdmin - LoadBalancer
    -   App - LoadBalancer
    -   Websocket - LoadBalancer
-   HorizontalPodAutoscaler
    -   Minimum: `1` replica
    -   Maximin: `3` replicas
    -   Metric: `CPU` usage `50% avg`
-   Network policy - network traffic only within namespace or via Ingress
-   Ingress
    -   Using built in `nginx`
    -   App available from host `python.app`
    -   PHP MyAdmin available from host `pma.python.app`

# Running

## Building app docker image

Before deploying app image needs to built and uploaded to. In `Dockerized` folder run following command

```sh
docker build -t python-app:latest -f dockerfile .
docker docker image save -o python-app.img python-app:latest
minikube image load python-app.img
```

Image upload process does not use normal `minikube image build` because of huge image size and memory constraints.

## Building websocket app docker image

Before deploying websocket app image needs to built and uploaded to. In `Dockerized/websocket` folder run following command

```sh
docker build -t websocket-app:latest .
minikube image load websocket-app:latest
```

In this case image is relatively small and can be loaded into minikube without creating file first.

## Preparing Minikube

```sh
minikube start
minikube addons enable ingress
minikube addons enable metrics-server
minikube addons enable registry
```

## Logs

To see everything running in `dockerized` namespace use following command.

```sh
kubectl get all,cm,secret,ing -n dockerized
```

To see more details run this command in new terminal window and keep it open.

```sh
minikube dashboard
```

## Deploying on Kubernetes

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
kubectl apply -f 10-app-hpa.yaml
kubectl apply -f 10.5-websocket.yaml
kubectl apply -f 11-ingress.yaml
kubectl apply -f 12-network.yaml
```

## Accessing

To obtain access to Kubernetes network open new terminal and run this command.

```sh
minikube tunnel
```

### Via Ingress

Firstly modify `hosts` file and paste the following lines, then substitute IngressIP from logs command of `ingress.networking.k8s.io/dockerized-ingress` resource

```txt
<IngressIP> python.app
<IngressIP> pma.python.app
```

Then open web browser and go to [app](http://python.app) or [PHP MyAdmin](http://pma.python.app)

# Cleaning up Kubernetes

To delete all created resources invoke this command.

```sh
kubectl delete namespace dockerized
```
