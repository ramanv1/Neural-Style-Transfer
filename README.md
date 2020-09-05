# Neural-Style-Transfer

We create a class to perform Neural Style Transfer. The class can make use of two pre-trained models viz. VGG19 and VGG16. We can set our style layers, content layers, change the weights of style loss, content loss and total variation loss. We also have the option of working with different optimizers such as Adam, RMSprop and SGD.
We demonstrate the use of the class in a jupyter notebook. 

We also provide the Dockerfile to make use of this class. 

In the folder containing the Dockerfile: docker build -t tf_dev .

Once the image is built: docker run -it --name=devTF -p 8888:8888 tf_dev

Then go to your browser and type: localhost:8888, you should see jupyter lab open.

Note: tf_dev is the docker image name, devTF is container name, 8888 is the exposed port on the container, your web UI port is 8888 on the host.

