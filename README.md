# ElasticRR

## Software Requirements
Python 3.7 and Pytorch 1.5.1 are used for the current codebase. 
We recommend you to create an environment and install necessary packages (e.g., numpy, pickle, pandas, codecs, etc.)
We ran our experiment on NVIDIA GeForce Titan Xp (GPU) or Intel Xeon E5-2637 v4 @ 3.50GHz (CPU), Linux (Ubuntu 16.04), CUDA 10.0, CuDNN 7.6.0.

#### Installation
- Create a virtual environment with conda/virtualenv
- Download the code folder
- Run: ```cd <PATH_TO_THE_CLONED_CODE>```
- Run: ```pip install -e .``` to install necessary packages and path links.

## Reproduce Paper Results
All the code of the paper are in `src/` folder. Dataset will be provided in the revision process. 
The repository comes with instructions to reproduce the results in the paper or to train the model with your dataset:

To run the default settings and reproduce the results: Go to folder `src/`
+ For getting the embedded features, please follow the instruction in the following link:
For text datasets: https://bert-as-service.readthedocs.io/en/latest/section/get-start.html#installation
For image datasets: https://github.com/christiansafka/img2vec

+ For randomizing inputs with a RR mechanism, e.g., f-RR, LATENT, OME, DM, PM, etc.: 
Run `python3 gen_fRRdata.py`. 
According to the dataset and the RR mechanism that you want, change line 223 (for the dataset) and line 241 (for the RR mechanism). 
The default code is for the AG dataset and f-RR mechanism.

+ For adding the label-DP or Laplace mechanism into the RR mechanism:
Run `python3 add_label_xx.py`, where `xx` is the name of dataset that you want to run, such as `ag` or `CelebA` or `sec`. 
For the FEMNIST data, this part is incorporated into the code for the classification task.  

+ For classification tasks: 
Run `python3 xx_input.py` if you want to perform the case of randomizing the embedded features, where `xx` is the name of dataset that you want to run. 
Run `python3 xx_grad.py` if you want to perform the case of randomizing the gradients without an anonymizer, where `xx` is the name of dataset that you want to run.
Run `python3 xx_ldpfl.py` if you want to perform the case of randomizing the gradients with an anonymizer, where `xx` is the name of dataset that you want to run.
Inside each file, you can choose different epsilon_X (and epsilon_Y) based on the privacy budget that you want.

+ There are several hyper-parameters that you can tune to achieve a good result, such as learning rate, number of hidden neurons, etc.

+ Note: Due to the privacy requirements of SEC data, this repository does not provide data and code for the SEC dataset. 

Enjoy the code! 

