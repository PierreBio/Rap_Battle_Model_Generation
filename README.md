![273459046-aa89660a-ef7d-4f2f-8318-f0fcf84e2623](https://github.com/jaimeMontea/MS_IA_NLP/assets/45881846/9b53a14e-cc23-4425-ae88-78be681b3275)

# Project Overwiew

This project focuses on the creation of a __rap battle model__ for the course of __Natural Language Processing__ of the  __Artificial Intelligence__ master's program at  __Telecom Paris__. The main objective of this work is to provide a complete approach to develop a rap battle model.

## Environment Setup

### Prerequisites

It's recommended to choose one workspace among the next ones to setup and achieve our work process:

- Google Colab account
- Kaggle account

*Important note: Some files are not notebooks. Make sure to convert them into notebook to use it on online workspace like Google Colab.*

## Steps to develop a model

*Important note: All the following steps has to be executed in the right order because you'll need to use the output of previous steps to achieve the current one.*

### 1. Web scrapping

#### Retrieve all the data from sources

In order to retrieve all the necessary battle to train our model, you have to launch the *scrape_data()* function contained in __web_scrapping.py__. 

We recommend to get all the battles in several times by modifying the values inside the next line for the *for* loop:

```
for battle_id in range(1, 49001):  
```

### 2. Data gathering

#### Put the retrieved data into database

Once all the data are scraped, you can execute the *gathering_data.py* file to put all the data into a database.

### 3. Data preprocessing

#### Clean the data and optimize it for the training

You can now use your generated database within the following notebook *data_cleaning.ipynb*. Please follow the instruction inside the file in order to correctly clean the data. Some functions has to be launched each time and some are not. 

*Important note: You have two parts that generate an output.*

- The before last part will generate 3 files, and especially __corpus_cleaned_lyrics__ that will be used at **Step 4**.
- The last part of this file will generate **corpus_cleaned_lyrics_batch_{batch}.txt** that will be used at **Step 6**.

### 4. Model training 

#### Use specific algorithms to train our model without taking into account Defender | Challenger architecture

Now, you should have a new generated file *corpus_cleaned_lyrics.txt*. You will use this file into *model.py* that you can convert into notebook if you prefer to use online workspace. You will generate three *.txt* files that are required to train and validate your new model. As an output, you should get a new model.

### 5. Model analysis

#### Play with hyperparameters and check efficience of model (accuracy, loss)

The *analysis.ipynb* is a specific file dedicated to show different analysis conducted onto our models. You can use it on your generated models to retrieve some data.

### 6. Model optimization 

#### Whole process to optimize all the previous steps to create a model with Defender | Challenger architecture

*Model_LSTM_optmized.ipynb* is the last file used in this project. It's dedicated to generate a specific model taking into account the rap battle mecanics with Challenger and Defender lyrics. To use this file, you have to locate your previously generated files *corpus_cleaned_lyrics_batch_{batch}.txt* at **Step 3** and use them.

## Important Notes

Our battle sources are taken from RapPad.co and are undoublty not under *Creative Commons License*. Therefore, you cannot use this work for any commercial project as long as you use these battles as sources.

## Credits

RapPad.co and all contributors of this website.
