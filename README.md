# Getting Started
 - Download the "Kaggle Dogs and Cats Redux" dataset
 - unzip `train.zip` and place the `cat.####.jpg` and `dog.####.jpg` files into `./data/train/`.
 - unzip `test.zip` and place the `####.jpg` files into `./data/test`.
 - Execute `run.sh`
 - To test your model, execute `evaluate.sh`

#### `run.sh` (`train.py`)
![Ayy, It's as easy as that bay-beeeee](https://i.imgur.com/DJ4vMPd.png)

#### `evaluate.sh` (`check.py`)
![TensorFlow really outdid themselves this time...](https://i.imgur.com/lLEUEge.png)

### Layout
 - `training.py` - Executes the network training.
 - `model.py` - Contains the network layer definitions.
 - `input_data.py` - Ingresses training data and creates lists and batches.
 - `check,py` - Tests the created model against one image, outputs prediction.

### Dependencies
`train.py`:
 - Python3
 - TensorFlow

`check.py`:
 - xvfb
 - python3-matplotlib

### Kaggle Dogs and Cats Redux dataset
 - Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data


# Additional Help

### Installing TensorFlow System Wide (No Virtual Environment) (Debian/Ubuntu)
```
sudo apt-get install python3-pip python3-dev
sudo pip3 install -U pip
sudo pip3 install -U tensorflow
```

### Install Xvfb and matplotlib (For headless image prediction tests (`check.py`)
```
sudo apt install python3-matplotlib
sudo apt install xvfb
```
