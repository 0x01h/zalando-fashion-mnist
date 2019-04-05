# zalando-fashion-mnist
My solution for "Zalando's Fashion MNIST" challenge on dockerized Jupyter Notebook.<br><br>
Before starting, download dataset to your work directory, extract it, and make sure you have `fashion-mnist_test.csv` and `test fashion-mnist_train.csv` files in project directory.

**Dataset download link:** https://www.kaggle.com/zalando-research/fashionmnist/downloads/fashionmnist.zip/4

**Note:** You have to be logged on Kaggle to download dataset.

### Example Project Directory
```
.
├── zalando-fashion-mnist
|   ├── fashion-mnist_test.csv
|   ├── fashion-mnist_train.csv
|   ├── build.sh
|   ├── .gitignore
|   ├── Dockerfile
|   ├── LICENSE
|   ├── README.md
|   ├── helpers.py
|   ├── main.py
|   ├── main.ipynb
```

### Installation & Usage
```
$ git clone https://github.com/0x01h/zalando-fashion-mnist.git
$ cd zalando-fashion-mnist
$ sh build.sh
```

1. Copy **Jupyter Notebook** session token that you are able see it in terminal.
2. Go http://127.0.0.1:8888 from your browser and sign in using token.
3. Open **work** directory.
4. Run **main.ipynb**.
