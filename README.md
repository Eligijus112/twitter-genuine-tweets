# Twitter NLP for determining whether a tweet is about a natural disaster or not

This project can serve as a template to do any NLP task. The sentiment in this project is whether a string regards a natural disaster or not.

The model is a bidirectional RNN. 

# Activating a virtual environment

For the creation of virtual environments (VE) anaconda is used. You can download anaconda via here: 
https://www.anaconda.com/distribution/

After installing anaconda into your system follow the steps bellow to work in a virtual environment.

Creation of the VE:
```
conda create python=3.7 --name nlp
```

Activating the VE:
```
conda activate nlp
```

Installing all the packages from the **requirements.txt** file to the virtual environment:
```
pip install -r requirements.txt
```

If you are using Microsoft Visual Studio code there may be some additional pop ups indiciating that some packages should be installed (linter or ikernel). 

# Word embeddings 

This project relies heavily on the pre trained word embeddings that are available here: https://nlp.stanford.edu/projects/glove/. The embeddings are stored in a .txt file and the file should be saved in the **embeddings/** directory. 