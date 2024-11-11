import os
import sys
import cv2
import time
import pickle
import numpy as np
from utils import *
import pandas as pd
from bow import BoW
from tqdm import tqdm
from dataset import Dataset
import matplotlib.pyplot as plt
from image_classifier import ImageClassifier


uploaded = False # Upload a charged df in case it exists


def descriptor_extrator(training_path:str, extractor_type:str = "sift") -> tuple[list, float]:
    """Extracts the descriptors from a set of images using the specified method.
    
    Args
    ----

    - training_path: Path to the folder containing the training images.
    - extractor_type: Type of descriptor extractor to use (shift/kaze).
        
    Returns
    -------
    
    - List of descriptors. 
    - Time taken to extract the descriptors.
    """
    
    # Load the data
    training_set = Dataset.load(training_path, "*.jpg")
    print(f"Loaded {len(training_set)} images.")
    
    t0 = time.time()
    
    if extractor_type == "sift":
        extractor = cv2.SIFT_create() 
        
    elif extractor_type == "surf": # Optimal time sift alternative
        extractor = cv2.xfeatures2d.SURF_create()
        
    elif extractor_type == "kaze":
        extractor = cv2.KAZE_create()
        
    elif extractor_type == "akaze": # Optimal time kaze alternative
        extractor = cv2.AKAZE_create()
        
    elif extractor_type == "brisk":
        extractor = cv2.BRISK_create()
        
    elif extractor_type == "fast": # Better for real time aplications
        cv2.FastFeatureDetector_create()
        
    print(f"Computing {extractor_type}...")
    time.sleep(0.1) # Prevents overlapping messages
    descriptors = []
    
    for image_path in tqdm(training_set, unit = "image", file = sys.stdout):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        try:
            _, descriptor = extractor.detectAndCompute(image, None) # Compute the descriptor
            
        except:
            print(f"WARN: Issue generating descriptor for image {image_path}")

        if descriptor is not None:
            descriptors.append(descriptor)
    
    return descriptors, time.time() - t0

def vocabulary_creation(descriptors: list, vocabulary_size: int, iterations: int, extractor_type:str ,output_path: str = "data/processed_data/vocabularies") -> tuple[str, float]:
    """Creates a vocabulary of visual words using K-means clustering.

    Args
    -----
    
    - descriptors: List of descriptors.
    - vocabulary_size: Number of visual words.
    - iterations: Number of iterations for the K-means algorithm.
    - extractor_type: Type of descriptor extractor used to generate the descriptors (shift/kaze).
    - output_path: Path to save the vocabulary.
    
    Returns
    -------
    
    - vocabulary_path: path to the saved vocabulary.
    - Time taken to create the vocabulary."""
    
    termination_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, iterations, 1e-6)
    words = cv2.BOWKMeansTrainer(vocabulary_size, termination_criteria)

    t0 = time.time()
    
    [words.add(descriptor.astype(np.float32))for descriptor in descriptors] # Add the descriptors to the trainer
    
    time.sleep(0.1) # Prevents overlapping messages
    print("\nClustering descriptors into", vocabulary_size, "words using K-means...")
    
    vocabulary = words.cluster()
    
    t1 = time.time()
    
    os.makedirs(output_path, exist_ok = True) # Create the output folder if it does not exist
    vocabulary_path =  os.path.join(output_path, f"{extractor_type}_{vocabulary_size}_{iterations}_vocabulary.pickle")
    
    # TODO: Open the file from above in the write and binay mode
    with open(vocabulary_path, "bw") as f:
        pickle.dump([extractor_type.upper(), vocabulary], f, pickle.HIGHEST_PROTOCOL)
        
    return vocabulary_path, t1 - t0
        
def train_classifier(vocabulary_path:str, training_path:str, vocabulary_size: int, iterations: int, extractor_type:str, output_path:str = "data/processed_data/classifiers") -> tuple[str, float]:
    """Trains an image classifier using the Bag of Words model.
    
    Args
    ----
    
    - vocabulary_path: Path to the vocabulary.
    - training_path: Path to the training set.
    - vocabulary_size: Number of visual words.
    - iterations: Number of iterations for the K-means algorithm.
    - extractor_type: Type of descriptor extractor used to generate the descriptors (shift/kaze).
    - output_path: Path to save the classifier.
    
    Returns
    -------
    
    - classifier_path: Path to the saved classifier files.
    - Time taken to train the classifier."""
    
    bow = BoW()
    
    bow.load_vocabulary(vocabulary_path.replace(".pickle", "")) # Access the vocabulary
        
    training_set = Dataset.load(training_path, "*.jpg") # Access the data set
    image_classifier = ImageClassifier(bow) # Load the vocabulary to the classifier
    
    t0 = time.time()
    
    image_classifier.train(training_set) # Train the classifier
    
    t1 = time.time()
    
    os.makedirs(output_path, exist_ok = True) # Create the output folder if it does not exist
    classifier_path = os.path.join(output_path, f"{extractor_type}_{vocabulary_size}_{iterations}_classifier")
    
    image_classifier.save(classifier_path) # Save the classifier
    
    return classifier_path, t1 - t0

def evaluate_classifier(vocabulary_path:str, classifier_path:str, test_path:str, save_data:bool = False, output_path:str = "data/processed_data/matrixes") -> tuple[float, float]:
    """Evaluates the performance of the classifier on a test set.
    
    Args
    ----
    
    - vocabulary_path: Path to the vocabulary.
    - classifier_path: Path to the classifier.
    - test_path: Path to the test set.
    - save_data: Parameter to save the confussion matrix.
    - output_path: Path to save the results.
    
    Returns
    -------
    
    - accuracy: Accuracy of the classifier.
    - Time taken to evaluate the classifier."""
    
    bow = BoW()
    bow.load_vocabulary(vocabulary_path.replace(".pickle", "")) # Load the vocabulary
    
    t0 = time.time()
    
    image_classifier = ImageClassifier(bow) # Load the classifier
    image_classifier.load(classifier_path) # Load the classifier
    
    test_set = Dataset.load(test_path, "*.jpg") # Load the test set
    accuracy, confussion_matrix, _ = image_classifier.predict(test_set, save = False) # Evaluate the classifier
    
    t1 = time.time()
    
    if save_data:
        os.makedirs(output_path, exist_ok = True)
        
        matrix_name = vocabulary_path.split("/")[-1].replace("vocabulary.pickle", "") + "confussion.csv"
        matrix_path = os.path.join(output_path, matrix_name)    
        
        np.savetxt(matrix_path, confussion_matrix, delimiter = ",")
        
    return accuracy, t1 - t0

def save_results(extractor_type:str, vocabulary_size:int, iterations:int, accuracy_train:float, accuracy_test:float, time_descriptor:float, time_vocabulary:float, time_classifier:float, time_evaluation_train:float, time_evaluation_test:float, output_path:str = "data/processed_data"):
    """Saves the results of the experiment to a DataFrame and a CSV file.
    
    Args
    ----
    
    - extractor_type: Type of descriptor extractor used to generate the descriptors (shift/kaze).
    - vocabulary_size: Number of visual words.
    - iterations: Number of iterations for the K-means algorithm.
    - accuracy_train: Accuracy of the classifier on the training set.
    - accuracy_test: Accuracy of the classifier on the test set.
    - time_descriptor: Time taken to extract the descriptors.
    - time_vocabulary: Time taken to create the vocabulary.
    - time_classifier: Time taken to train the classifier.
    - time_evaluation_train: Time taken to evaluate the classifier on the training set.
    - time_evaluation_test: Time taken to evaluate the classifier on the test set."""
    
    global df, uploaded
    
    if not uploaded:
        try:
            df = pd.read_csv(os.path.join(output_path, "results.csv"))
            uploaded = True
            
        except: # Create a new dataframe in case it does not exist
            df = pd.DataFrame(columns = ["Extractor", "Vocabulary size", "Iterations", "Accuracy train", "Accuracy test", "Time descriptor", "Time vocabulary", "Time classifier", "Time evaluation train", "Time evaluation test"])
        
    df.loc[len(df)] = {"Extractor": extractor_type, "Vocabulary size": vocabulary_size, "Iterations": iterations, "Accuracy train": accuracy_train, "Accuracy test": accuracy_test, "Time descriptor": time_descriptor, "Time vocabulary": time_vocabulary, "Time classifier": time_classifier, "Time evaluation train": time_evaluation_train, "Time evaluation test": time_evaluation_test}
    
    os.makedirs(output_path, exist_ok = True)
    
    df.to_csv(os.path.join(output_path, "results.csv"), index = False)
    
if __name__ == "__main__":
    
    training_path = "data/dataset/training"
    test_path = "data/dataset/validation"
    
    vocabulary_sizes = [100, 200, 400]
    iterations = [10]
    extractor_types = ["kaze"] # "sift", "surf", "kaze", "akaze", "brisk"
    # Surf is not publicly available in the current version of OpenCV, akaze and brisk do not work well with the image_classifier.py file
    for extractor_type in extractor_types:
        for vocabulary_size in vocabulary_sizes:
            for iteration in iterations:
                
                print(f"\n\nRunning experiment for {extractor_type} with {vocabulary_size} words and {iteration} iterations...")
                
                descriptors, time_descriptor = descriptor_extrator(training_path, extractor_type)
                vocabulary_path, time_vocabulary = vocabulary_creation(descriptors, vocabulary_size, iteration, extractor_type)
                classifier_path, time_classifier = train_classifier(vocabulary_path, training_path, vocabulary_size, iteration, extractor_type)
                accuracy_train, time_evaluation_train = evaluate_classifier(vocabulary_path, classifier_path, training_path)
                accuracy_test, time_evaluation_test = evaluate_classifier(vocabulary_path, classifier_path, test_path)
                
                save_results(extractor_type, vocabulary_size, iteration, accuracy_train, accuracy_test, time_descriptor, time_vocabulary, time_classifier, time_evaluation_train, time_evaluation_test)
                
                print(f"Experiment for {extractor_type} with {vocabulary_size} words completed.")
            
    print("\n\nExperiment completed.")
    