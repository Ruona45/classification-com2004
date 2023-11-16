"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""

from typing import List

import numpy as np
import scipy.linalg

N_DIMENSIONS = 10

""" Helper Functions """
def euclidean_distance(x1,x2):
     """Calculate the Euclidean distance between two points."""
     value = np.sqrt(np.sum((x1-x2)**2))
     return value

def find_k_nearest_neighbours(training_data: np.ndarray, test_data: np.ndarray, k: int):
    distances = []
    neighbours = []
    
    for train_data in training_data:
        distance = euclidean_distance(test_data, train_data)
        distances.append((train_data, distance))

    # Sort distances in ascending order to find the nearest neighbors
    distances.sort(key=lambda x: x[1])

    # Select the k nearest neighbors
    for i in range(k):
        neighbours.append(distances[i][0])

    return neighbours
## With K=3
# """ Running evaluation with the clean data.
# Square mode: score = 70.8% correct
# Board mode: score = 70.8% correct
# Running evaluation with the noisy data.
# Square mode: score = 57.6% correct
# Board mode: score = 57.6% correct  """


## WORK ON THIS FUNCTION
def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray, k=6) -> List[str]:
    ## Add the k value with the default of 3 for now
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.
        k (int): Number of nearest neighbors to consider 

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    
    labels = []
    ## Set a default value of K=3 for now will do test later to pick a better value
    
    for i in range(test.shape[0]):
        test_instance = test[i]
        neighbours = find_k_nearest_neighbours(train,test_instance,k)
        neighbour_indices = [np.where((train == neighbour).all(axis=1))[0][0] for neighbour in neighbours]
        neighbour_labels = [train_labels[idx] for idx in neighbour_indices]
        predicted_label = max(set(neighbour_labels), key=neighbour_labels.count)
        labels.append(predicted_label)
    
    return labels

# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.

## WORK ON THIS FUNCTION
def reduce_dimensions(data: np.ndarray, model: dict, n_components=N_DIMENSIONS) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    # If already computed PCA use those values 
    if "meanVals" in model and "redVects" in model:
        meanVals = np.array(model["meanVals"])
        reducedVects = np.array(model['reducedVects'])
        centered_data = data - meanVals
        reduced_data = np.dot(centered_data,reducedVects)
    else: 
        # Calculate the mean of the data along each feature dimension
        meanVals = np.mean(data, axis=0)
        centered_data = data - meanVals
        
        # Get the cov matrix
        covx = np.cov(centered_data,rowvar=0)
        
        # Get the eigenvalues and eigenvectors
        eigVals, eigVects = scipy.linalg.eigh(covx)
        
        # Sort eigenvectors in desceding order
        idx = np.argsort(eigVals)[::-1]
        eigVects = eigVects[:, idx]
        
        reducedVects = eigVects[:, :N_DIMENSIONS]
        reduced_data = np.dot(centered_data,reducedVects)
        
        # Store mean and eigenvectors in the model for future use
        model["meanVals"] = meanVals.tolist()
        model["reducedVects"] = reducedVects.tolist()
    
    return reduced_data

## WORK ON THIS FUNCTION
def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.
    model = {}
    model["labels_train"] = labels_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors

## WORK ON THIS FUNCTION
def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    return classify_squares(fvectors_test, model)
