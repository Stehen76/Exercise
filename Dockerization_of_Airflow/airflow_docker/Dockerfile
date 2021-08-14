FROM apache/airflow:2.1.2
# WORKDIR <name-of-working-directory-in-container> - Defines the home directory of your container
WORKDIR /app
# Copy requirements file from the directory the Dockerfile is in into the /app directory of the container
COPY requirements.txt /app/requirements.txt
COPY iris.csv /app/iris.csv
#RUN apt-get -y update
#RUN apt-get -y install git
#RUN git clone https://github.com/mlflow/mlflow.git
RUN pip install -r requirements.txt
