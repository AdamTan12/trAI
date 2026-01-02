## trAI
Trash classifier and locater

## Required Packages
Install all dependencies using pip:

# bash
pip install torch torchvision torchaudio numpy matplotlib Pillow

# recommended specs
A100 on Google Colab to retrain

# if using google colab
download training data into google drive
![alt text](/readme%20screenshots/image-5.png)
![alt text](/readme%20screenshots/image-6.png)
# mount drive onto colab and download training data
![alt text](/readme%20screenshots/image.png)
![alt text](/readme%20screenshots/image-4.png)

# training loop
python train.py
![alt text](/readme%20screenshots/image-2.png)
will download weights every 10 epochs

# run the model
python test.py
![alt text](/readme%20screenshots/image-3.png)

# change input image when running the model
download an image, put it in current folder
edit path name on line 21 in test.py
![alt text](/readme%20screenshots/image-1.png)
