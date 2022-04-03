# Training Retinanet Model on VisDrone Dataset 


### Setup
Install dependencies
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install pandas matplotlib
pip install opencv-python sklearn pycocotools requests
sudo apt-get install tk-dev python-tk
```

Install VisDrone dataset
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn" -O VisDrone2019-DET-train.zip && rm -rf /tmp/cookies.txt
unzip VisDrone2019-DET-train.zip
```

Setup classes
```
python make_anno.py
```

### Training

```
python train.py --dataset csv --csv_train visDrone_train.csv --csv_classes classes.csv --csv_val visDrone_valid.csv
```


### Visualize
`--model_path` is a `.pt` file. Use `model_final.pt` either that has just trained or the one in Google Drive.
```
python visualize_single_image.py --image_dir test_images --model_path model_final.pt --class_list classes.csv
```


