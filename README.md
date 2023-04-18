# Project description
Classify the rat's lever-press state (press/rest/nan) from his neural firing information (spike).

The project report can be accessed via https://drive.google.com/file/d/1_3MDYz6PKgYMvz8cA729wtkWJjWC-IOG/view?usp=share_link

# How to run my code?
Make sure the following packages are installed in the environment:

python 3.8.8

pytorch 1.8.1

numpy

scipy

scikit-learn

tqdm

mat4py

Training command ```>>python main.py```

Testing command ```>>python test.py```

# I can't run >>python main.py
main.py embeds both 0/1 prediction and nan/valid prediction. It is set to 0/1 prediction as default.

If run 0/1 prediction, please comment all modules labelled nan/valid prediction.

If run nan/valid prediction, please comment all modules labelled 0/1 prediction.

# I can't run >>python test.py
test.py is set to 0/1 prediction as default.
