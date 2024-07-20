Exploration of PCA and SVD on MNIST Data
=========
This holds the code and backing for the third assignment of the EE399 class. The main dataset this assignment revolves around is the MNIST 784 which is a set of 70000 28x28 images of digits. These digits are processed in order to be categorized by SVD and PCA methods. They are also fed into SVM and Decision tree models in order to compare accuracies between these models and how they handle this dataset. 

Project Author: Elijah Reeb, elireeb@uw.edu

.. contents:: Table of Contents

Homework 3
---------------------
Introduction
^^^^^^^^^^^^
This assigment consists of two main parts. The first involves determining and understanding how Single Value Decomposition (SVD) works and applying it to the MNIST dataset. The second part involves the use of a 4 component Principle Component Analysis (PCA) and applying this dimension reducing method to the data set in order to more efficiently process the data with Support Vector Machine (SVM) and Decision Tree models. These four different models for interacting with data will be explained and broken down in this document with each subpart commenting on each of the four models. 

Theoretical Backgroud
^^^^^^^^^^^^
The theoretical explanations for how these models work are all fairly simple. The SVD and PCA models involve linear algebra tricks in order to use matrix properties to determine relationships between parts of data. 
First, the SVD model. The SVD model revolves around determining 3 main matricies that will create a space to map the data onto in order to reduce the dimensions of the data. This diagram from wikipedia helps visualize:

.. image:: https://user-images.githubusercontent.com/130190276/234205018-f10e3564-15c3-410e-b52f-caf0494845ca.png

The M matrix (also sometimes refered to as A) holds the dataset. In this case it is 70000 by 784 where each image is vectorized and stored. In an SVD model the U, Σ, and V matricies are the 3 main parts that the data is broken down into. The U matrix is the size of the data x the size of the data. In this case this is a very large 70000x70000 matrix and is unitary to hold all the data points. The Σ matrix is a diagonal matrix that holds the values of each of the data points in U to get them towards V. Last V holds the rank of the number of components used in the SVD model. In the case above, the SVD was done with 70 components so this is the space that the SVD now exists in and where the main dimensionality is reduced. 

Next, the PCA model. The PCA model is similar to the SVD model; however, the PCA model assumes that the data has been normalized with zero mean. This means that PCA focuses only on values that are more "relevant." This distinguishment will be shown in the similarity when the 4 PCA modes are plotted and can be visually compared to the 9 SVD modes. 

Next, the SVM model. The algorithm section will go more in depth to how the SVM algorthim works. Simply definied here, the SVM aims to draw lines between the data points. The Covers theorem implies that as infinite dimensions are added infinite data can be split. Due to computation limitations this will not go to infinity, but as the dimensions decreases, the SVM will become more effective at categorizing the data. The image below from wikipedia helps to visualize in a low dimension space:

.. image:: https://user-images.githubusercontent.com/130190276/234217160-7dc0630b-0257-4b00-805e-015f32f75889.png

Finally, the Decision Tree (random forest) model. The theory behind this method involves taking each parameter in a vector and determining which has the most impact and splitting the data based on that. Then this is done for the next parameter until all of the parameters are used to split the data. This method is somewhat costly on a large space but in this model where the PCA components were used it is very efficient. The image below from javatpoint.com helps to visualize:

.. image:: https://user-images.githubusercontent.com/130190276/234216982-233358a6-ee4c-400b-a885-0b2d32ebc2c1.png

Algorithm Implementation and Development
^^^^^^^^^^^^
With the refinement and sklearn toolkit in python the algorithm implementation is very straightfoward in python code. Continuing the format as above each of the main algorithms behind the models are explained with the simple code to implement them described as well. 

First, the SVD model. This can be implemented using the TruncatedSVD command with a set number of components as shown below. This number becomes r or the rank of the SVD space and thus greatly reduces the required dimensions for later classification. About 70 modes are where most of the SVD value is held with very little added benefit to more than that. This means the rank r of the digit space is reduced from 784 to 70 in order to more easily process the more valuable parts of the images. The algorithm is just using linear algebra to determine the matricies that multiply together in the order shown above. Once these matricies are found they can be applied to the data space in order to begin to classify. 

.. code-block:: text

        svd = TruncatedSVD(n_components=70)
        X_svd = svd.fit_transform(X)

The first 9 SVD modes are plotted below and can be noted as very similar to the 4 PCA modes plotted further below. These images are the main features in all of the number data and we can see that there are faint numbers in each of them and they can be combined to determine which input is which digit. 

.. image:: https://user-images.githubusercontent.com/130190276/234218570-9872280e-e9aa-49e6-9ad8-a96723e42363.png

These SVD modes can individually be applied to the data in order to map the data to the SVD space. Three chosen modes were applied to the data and plotted in 3D below. This shows that with very little linear algebra and only a few modes the data begins to "spread apart" and become more apparant to be classified. While the cluster is closely packed, we still can observe separation and different digits in different spaces. 

.. image:: https://user-images.githubusercontent.com/130190276/234225484-ccab29b2-b68b-4b6b-9d98-313e9b4e5924.png

Next, the PCA model. 

The PCA model follows the same algorithm as above. It just uses more preprocessing of the data to have zero mean. In this method there were only 4 components used to transform the data. 

.. code-block:: text
        
        pca = PCA(n_components=4)
        X_pca = pca.fit_transform(Xnp)

The 4 PCA component modes are plotted below. They hold a large amount of what makes up the numbers. 

.. image:: https://user-images.githubusercontent.com/130190276/234219587-d2c5886c-f753-4395-bb71-eb5ee3bf84b5.png

This PCA space was used to create the data sets in order to run the SVM and decision tree models. Note that it is also important to section the data into a test set and a training set. This allows more validation of the models as in algorithms such as the decision tree model there will be 100% accuracy on the training data but that should not be interpreted as the test accuracy. 

.. code-block:: text

        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)


Next, the SVM model. This model is much harder to visualize, however it can be set up with just a few lines of code. The algorithm involves constantly aiming to draw hyperplanes between the dimensions of the data. The loss function involves minimizing the error between predicted output and actual output while also limiting the amount of dimensions used. 

.. code-block:: text

     clf = SVC()
     clf.fit(X_train, y_train)
     y_pred = clf.predict(X_test)


Finally, the Decision Tree (random forest) model. As described above, each parameter is taken and used as the sole way to sort the data. The parameter with the most effect is chosen first and the data is split. This is repeated until the data is completely split. Other methods of "random forest" involve splitting the data into different sections in order to gain different "trees" that can be compared. This leads to "bagging" and "boosting" which will be discussed in later repositories. Similar to the other methods, python makes this classification easy to implement with code like below. 

.. code-block:: text

        treeclf = DecisionTreeClassifier(random_state=42)
        treeclf.fit(X_train, y_train)
        y_pred = treeclf.predict(X_test)

Computational Results
^^^^^^^^^^^^
With all four of the tested models the accuracy was computed when applying the model to either the entire dataset of 10 digits or the model was applied to the most easy to separate digits (0 and 1) and the hardest (7 and 9).  

In order to determine which digits are easiest or hardest to separate a LDA classifier was applied to all combinations of the digits. The accuracy was then measured and is plotted below. This shows clearly which digits are similar in the PCA space and how easily this linear method can be used to classify. 

.. image:: https://user-images.githubusercontent.com/130190276/234224854-5fcb4298-6fdf-4726-abbd-dff0b7494fdf.png

This was first done using a LDA or linear classifier on the PCA components. This is where one space was mapped and a single line was drawn to separate the data. This is shown with the 0 and 1 data where it is very clear one line can be drawn to separate the data. This method had more than 99% accuracy on the training and test data which is expected because these numbers are "far apart" in PCA space. 

.. image:: https://user-images.githubusercontent.com/130190276/234222786-de67cdea-3fef-423b-b7ce-684b52ade85a.png

When a third digit (5) is added the accuracy decreases as it is harder to draw 2 lines to separate the data and there is higher overlap between the digits. Shown below. This still has close to 90% accuracy on test and training data. This shows this method is relatively sound to distinguish some digits. 

.. image:: https://user-images.githubusercontent.com/130190276/234223333-f1a83539-393e-460c-975a-0875448915c3.png

When this method is tried on the 7 and 9 digits we can observe how much overlap there is and thus an accuracy of closer to 56% was achieved. This is evidence for SVM or decision trees being necessary. 

.. image:: https://user-images.githubusercontent.com/130190276/234224040-0f6bbb71-1bd4-44c2-a54d-09458cb55e71.png

Transitioning to SVM and decision trees, these are harder to visualize and will not be plotted. These models can have their accuracies compared on the 0 1 and 7 9 comparisons. 

.. image:: https://user-images.githubusercontent.com/130190276/234226088-ce8af86e-9b5a-4f91-9831-1336eec55267.png

These results will be discussed further in the conclusions. 

Summary and Conclusions
^^^^^^^^^^^^
Comparing the accuracy between the 3 models on training and test data, the decision tree had the highest on the training data with 100% accuracy. This is thought to be because the tree breaks down the training data to the individual branches so each data point is given what it is. The 3 models had comparably high results for the testing data on the 0 and 1 classification all over 99%. When looking at distinuishing 7 and 9 the decision tree had 100% accuracy for the training set for the reason stated above, however all three models were much lower on the testing set. The SVM model was the highest with 70% compared to 56% and 63% for the LDA and decision tree respectively. The SVM with this result is probably the best model all around. The SVM model also had a somewhat high accuracy when applied to the whole mnist data set. Connecting to the class discussion, when these models are ran on the mnist dataset itself they are much more accurate than when run on the PCA space of the mnist data set. This is due to a trade off of information for processing time. The PCA reduced the data to 4 main components in this case as opposed to the potential 784. This means it would have taken much longer to compute these models on the data without this middle step of PCA. 

In summary, there was a clear tradeoff between computing time and accuracy in this assignment. As more time was taken to perform more calculation, more accuracy was reached. The ability to first lower the dimensions of a data set to its PCA or SVD modes is a good way to save time. One of the larger computation processes was the SVM as despite the number of components in the PCA space, the SVM can increase dimensions to aim for better classification. This dataset was 70,000 and when looking at much larger data sets, these reduction techniques will become increasingly valuable. As talked about in the code sections, the code to implement these models is very simple so it is on the coder to understand what they are doing in order to reduce some of the "black box" that Machine Learning is. 
