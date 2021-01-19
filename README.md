# Running Locally on Your Laptop

```shell
# For this app, you need to have Docker installed on your PC. Make sure your Docker is up and running
#Open terminal or command prompt and execute scripts 
git clone 'my repo by taking the HTTPS or SSH link' && cd eda-using-dash
# This command is to stop is the container is already up and running
docker stop dash_app 
# This command is to remove the container if it already exists
docker rm dash_app
# This command is to build image
docker build --rm -t dash_app:latest .
# This command is to run the build container
docker run -p 8050:8050 --name dash_app dash_app:latest
# You would see that a message that a flask app is running on http://0.0.0.0:8050/
# After this open any browser and go to http://localhost:8050/

```
