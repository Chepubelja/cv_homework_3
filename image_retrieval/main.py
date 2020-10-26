"""
Module with the main logic.
"""

from tqdm import tqdm

from image_retrieval.utils import *


if __name__ == '__main__':
    PATH_TO_DATA = "data"

    QUERIES = ["ukbench00004.jpg",
               "ukbench00040.jpg",
               "ukbench00060.jpg",
               "ukbench00588.jpg",
               "ukbench01562.jpg"]

    NUM_MATCHES_TO_PLOT = 10

    all_hists = {}

    # Task 1 - Calculating histograms for all 2000 images and storing results in `all_hists` dictionary
    for image_name in tqdm(os.listdir(PATH_TO_DATA), position=0):
        path_to_image = os.path.join(PATH_TO_DATA, image_name)
        image = read_rgb_image(path_to_image)
        hists = calculate_hist_per_channel(image)

        all_hists[image_name] = hists

    for query in QUERIES:
        query_hists = all_hists[query]

        # Task 2 - Measuring L2 distance between query's and all images' histograms
        distances = [(img_name, l2_distance(query_hists, hists)) for img_name, hists in all_hists.items()
                     if img_name != query]
        sorted_dist = sorted(distances, key=lambda _: _[1])

        # Task 3 - Retrieving top-10 matches and plotting them
        # (+ storing them in `plots/matches` directory)
        matches = sorted_dist[:NUM_MATCHES_TO_PLOT]
        plot_matches(query, matches, PATH_TO_DATA)

        # Retrieving correct matches for each query
        correct_matches = []
        for i in range(1, 4):
            match_n = int(query.split('.')[0].split('ukbench')[1]) + i
            if match_n == 1562:
                match_n = 1560
            match_n_full = ''.join(['0' for _ in range(5 - len(str(match_n)))]) + str(match_n)
            match_correct = f"ukbench{match_n_full}.jpg"
            correct_matches.append(match_correct)

        # Task 4 - Measuring and plotting the Precision-Recall curve for each query
        # (+ storing them in `plots/curves` directory)
        precision = calculate_precision(sorted_dist, correct_matches)
        recall = calculate_recall(correct_matches)
        plot_pr_curve(query, precision, recall)

    # Task 5 - Discuss and explain success and failure cases

    """
    Let's start with success case:
    
    - 'ukbench00040' - First 3 matches are the correct ones.
    
    The main reason why I think this query has been successful in terms of matches retrieval is that
    the query image and its correct matches have very particular colors (or color histogram), 
    unlike others' images histograms. 
    Also, in terms of visual difference between this query image and its correct matches,
    the difference is some subtle affine transformation, that's why they are so similar.
    
    For the failure case lets take a look at this one:
    
    - 'ukbench01562' - Only third match (from first 10 retrieved matches) is correct.
    
    I believe that there is a simple explanation, why color histogram comparison doesn't work here.
    All the retrieved (first 10) matches have one thing in common:
    there is some object in the center lying on the table. 
    The problem here is that roughly half of the image (counting by pixels) is taken by table.
    So, histogram comparison method is failing for these images, 
    cause it's highly sensitive for the pixel values (in our case - values for the table color, which take big share).
    """