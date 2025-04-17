FROM jupyter/base-notebook:latest

# Set the working directory
WORKDIR /home/jovyan

# Copy the environment.yml file into the container
COPY requirements.txt .

# Install the Conda environment from the YAML file
RUN pip install -r requirements.txt

# Set the default command to start JupyterHub
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.ip='0.0.0.0'", "--NotebookApp.port=8888"]