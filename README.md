# Comparing Food-101 classification by Transformer or LSTM
This repository compares the performace of two deep learning architectures, DistilBert transformer and LSTM, for classification on the [Food 101-Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). The Food-101 dataset comprises 250 images and captions of each of the 101 dishes.  We only use the text captions in our coomparison. The text files, `train_titles.csv` and `test_titles.csv`, are provided in the repository.  

# Files and description
There are two python code files in this repository, which are explained below: 

**1. distilBertClassifier.ipynb:** python notebook that imports the pre-trained distilbert transfomer from huggingface, fine-tunes the weights, and tests the fine-tuned model.

**2. lstmClassifier.ipynb:** python notebok that trains and tests the lstm + feedfoward network.

Please email any questions to aabbasi1@iastae.edu 
