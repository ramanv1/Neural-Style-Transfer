FROM tensorflow/tensorflow:latest


RUN apt-get update -y && \
    apt-get install -y python3-pip && \
    pip3 install jupyterlab==2.2.6 &&\
    pip3 install numpy &&\
    pip3 install pandas &&\
    pip3 install scikit-learn &&\
    pip3 install matplotlib

EXPOSE 8888

CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=
