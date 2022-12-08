import numpy as np
import random
import math
#
# Name: Jared Tanuwidjaja
# SID: 862173492
#
# DISCLAIMER:
# A large portion of boiler plate code has been used. Both feature_search() and use
# code provided by Professor Keogh in the Project 2 briefing. All significant code
# is original and has been implemented by myself.
#

def main():
    #print("Choose your data set:\n 1 - Small Data Set\n 2 - Large Data Set\n")
    choice = input("Choose your data set:\n 1 - Small Data Set\n 2 - Large Data Set\n")
    while (choice != '1' or choice != '2'):
        if (choice == '1'):
            print("\n   Loading in the small data set...    \n")
            arr = np.loadtxt("CS170_Small_Data__6.txt")
            print(arr.shape)
            break
        elif (choice == '2'):
            print("\n   Loading in the large data set...    \n")
            arr = np.loadtxt("CS170_Large_data__123.txt")
            print(arr.shape)
            break
        else:
            print("Please enter a valid input:\n    1 - Small Data Set\n    2 - Large Data Set\n")

    #Run feature_search on the data set
    feature_search(arr)

def leave_one_out_cross_validation(data, current_set_of_features, feature_to_add):
    number_correctly_classified = 0

    
    for i in range(0, data.shape[0]):
        object_to_classify = data[i,1:]
        label_object_to_classify = data[i,0]
        
        nearest_neighbor_distance = np.inf
        nearest_neighbor_location = np.inf
        
        for k in range(0, data.shape[0]):
            if k != i:
                for y in range(1, data.shape[1]):
                    if y not in current_set_of_features and y != feature_to_add:
                        data[k,y] = 0
                #print("Asking if " + str(i+1) + " is nearest neighbour with " + str(k+1))
                temp = np.subtract(object_to_classify, data[k,1:])
                distance = math.sqrt(np.dot(temp, temp))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location,0]
                
            
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified = number_correctly_classified + 1
        
    print(number_correctly_classified/data.shape[0])
    return number_correctly_classified / data.shape[0]

def feature_search(data):
    current_set_of_features = [] #Initialize empty set

    for i in range(0, data.shape[1]-1):
        print("On level "+str(i+1)+" of the search tree")
        feature_to_add_at_this_level = None
        best_acc_so_far = 0
        
        for k in range(0, data.shape[1]-1):
            if (k not in current_set_of_features):
                print("-- Considering adding the "+str(k+1)+" feature")
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k+1)
                
                if accuracy > best_acc_so_far:
                    best_acc_so_far = accuracy
                    feature_to_add_at_this_level = k

        current_set_of_features.append(feature_to_add_at_this_level)
        print("On level "+str(i+1)+" I added feature "+str(feature_to_add_at_this_level)+" to current set")
    print("Finished search! The best feature set is ",current_set_of_features)
    print(" It has an accuracy of ", best_acc_so_far)

if __name__ == "__main__":
    main()