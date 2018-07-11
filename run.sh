if ! [ -d "data" ]; then
echo "data directory does not exist"
echo "Download from https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data"
echo "Structure should look like, EX: ./data/train/cat.1234.jpg"
exit
fi
python3 ./training.py
