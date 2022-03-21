# Curiosity_project
This repo is for project that was done as part of Curiosity models and applications course by Goren Gordon at TAU.

In this project we learn the formulation of curiosity proposed by prof. Gordon in several papers, in particular (Gordon  et al., [2012](http://docs.wixstatic.com/ugd/80855c_e23ff5655de44647b2269b47dfaab589.pdf), Moran et al., [2019](https://docs.wixstatic.com/ugd/8c0099_cdde28eda25c472da31bd81a5cee5238.pdf)).

We use this approach in order to build and examine adversarial examples, we are mostly based on paper by Goodfellow  et al., [2014](https://arxiv.org/pdf/1412.6572.pdf)



### Directory sturcture


    .
    ├── current_notebooks                   # up-to-date jupiter notebooks with the code
    ├── data                    
    ├── doc                                 # documents
    ├── gephi_graphs                        # ready gephi graphs
    ├── images                    
    ├── networkx_graphs                     # graphs from networkx that are ready for gephi visualization
    ├── old_notebooks
    ├── Complex networks - project.pdf      # project report
    ├── README.md
    └── requierments.txt


### Project report 
Project report can be found [here](https://github.com/SolOlga/ComplexNetworks_project/blob/master/Complex%20networks%20-%20project.pdf)




### Important files:

    mnist_model.py - creates and trains a CNN for classification of handwritten digits (MNIST dataset) Note that a trained model mnist_cnn_epoch62.pt is supplied under the trained_models directory.
    NN_pert_gen.py - creates and trains a CNN for generation of adversarial examples (AE) based on input images.
    curiosity_loop.py - runs a curiosity loop and tests it.
    curiosity_loop_analysis.py - creates graph and compares learned policy with random policy.
    dataset.py - a Dataset object for inner use in curiosiy loop and related algorithms.

All the code assumes a windows operating system.

To run start with curiosity_loop.py and proceed to curiosity_loop_analysis.py. Running this may take over 40 hours (using i7 7700k processor and nvidia gtx 1080ti gpu) Note that under the trained_models folder you can find all the files creted by running these, except for trained models which are not required for the analysis.

some packages needed for running are torch, torchvision, numpy, matplotlib, scipy.
