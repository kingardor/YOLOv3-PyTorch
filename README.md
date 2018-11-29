# YOLOv3 Object Detector PyTorch Implementation

This repository is built upon [Ayoosh Kathuria's](https://github.com/ayooshkathuria) work.

## Requirements

* Python 3.6+
* PyTorch 0.4.0
* OpenCV 3.4.1+

## Creating a Virtual Environment for Python

Yes, using a virtual environment for python is better cause you wouldn't want your system python dependencies messed up.

```sh
# To install pip
sudo apt-get install python3-pip

# To install virtualenv and its wrapper
sudo pip3 install virtualenv virtualenvwrapper
```

Add the following lines to .bashrc (.zshrc if you use zsh) by running the following commands:

```sh
echo '# virtualenv and virtualenvwrapper' >> .bashrc
echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6' >> .bashrc
echo 'export WORKON_HOME=$HOME/.virtualenvs' >> .bashrc
echo 'source /usr/local/bin/virtualenvwrapper.sh' >> .bashrc
```

Creating a virtual environment:
```sh
mkvirtualenv nameofenv -p python3.6
```
Using a virtual environment:
```sh
workon nameofenv
```
Deactivating a virtual environment:
```sh
deactivate
```
## Download YOLOv3 Weights

Download the weights using the following command
```sh
wget -O yolov3.weights https://pjreddie.com/media/files/yolov3.weights
```
## Running Detection

Run the folllowing script to start the detector
```sh
python3 object.py
```

Hold on, there are some command line arguments you can utilize
* --confidence - To set the confidence level
* --nms_thresh - To set the NMS Threshold
* --reso - To set the input resolution
* --source - To set the input camera source
* --skip - To skip every alternate frame or not, for faster processing speed
