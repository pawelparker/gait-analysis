# Gait-analysis
Gait Analysis using Extreme Learning Machine (ELM) based classifier along with MLP, Ensemble, and SVM

*Human Gait Classification-1*

Given Accelerations(about x,y and z axis)  and Angular Velocites (about x,y and z axis) and Joint Angles in Degrees classfiy into 7 different Tasks:Jogging, Normal walk, sit up,  up stair walk,  down stair walk, walk on heel and walk on toe.

*Human Gait Classification -2*

Given Joint rotation in Degrees, Classify into 9 tasks: Walking Normal Speed, Walking Very Slow, Walking Slow, Walking Medium, Walking Fast, Walking on Toes, Walking on Heels, Staircase Ascending, Straincase Desending

# Extreme Learning Machine (ELM) implementation
The ELM algorithm is similar to other neural networks with 3 key differences:

1.The number of hidden units is usually larger than in other neural networks that are trained using backpropagation.

2.The weights from input to hidden layer are randomly generated, usually using values from a continuous uniform distribution.

3.The output neurons are linear rather than sigmoidal, this means we can use least square errors regression to solve the output weights.

For ELM package installation:

pip install git+https://github.com/masaponto/python-elm

Usage Example:

from elm import ELM

elm = ELM(hid_num=100000).fit(train_set, train_set_labels)
