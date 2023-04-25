UW-EE399-Assignment-3
=========
This holds the code and backing for the third assignment of the EE399 class. The main dataset this assignment revolves around is the MNIST 784 which is a set of 70000 28x28 images of digits. These digits are processed in order to be categorized by SVD and PCA methods. They are also fed into SVM and Decision tree models in order to compare accuracies between these models and how they handle this dataset. 

Project Author: Elijah Reeb, elireeb@uw.edu

.. contents:: Table of Contents

Homework 3
---------------------
Introduction
^^^^^^^^^^^^


Theoretical Backgroud
^^^^^^^^^^^^

.. code-block:: text

        nfaces = 100
        Xm=X[:, 0:nfaces]
        C=np.matmul(Xm.T,Xm)


.. image:: https://user-images.githubusercontent.com/130190276/232986883-1fd44f06-b268-4bcd-9758-db951b665604.png



Algorithm Implementation and Development
^^^^^^^^^^^^

.. code-block:: text

        Y=np.matmul(X,X.T)
        vals,vects = np.linalg.eig(Y)


Computational Results
^^^^^^^^^^^^


Summary and Conclusions
^^^^^^^^^^^^
