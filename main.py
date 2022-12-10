import numpy as np
import random
import math
import time
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
            arr = np.loadtxt("CS170_Small_Data__88.txt")
            break
        elif (choice == '2'):
            print("\n   Loading in the large data set...    \n")
            arr = np.loadtxt("CS170_Large_data__96.txt")
            break
        else:
            print("Please enter a valid input:\n    1 - Small Data Set\n    2 - Large Data Set\n")

    #Run feature_search on the data set
    choice = input("Choose your algorithm:\n 1 - Forward Selection\n 2 - Backwards Elimination\n")
    while (choice != '1' or choice != '2'):
        if (choice == '1'):
            print("\n   Running Forward Selection...    \n")
            forward_select(arr)
            break
        elif (choice == '2'):
            print("\n   Running Backwards Elimination    \n")
            backward_elim(arr)
            break
        else:
            print("Please enter a valid input:\n    1 - Forward Selection\n    2 - Backwards Elimination\n")

    

def leave_one_out_cross_validation(data, current_set_of_features, feature_to_add):
    number_correctly_classified = 0
    
    features = []
    features.append(0)
    features.append(feature_to_add)
    for x in range(0, len(current_set_of_features)):
        features.append(current_set_of_features[x])
    tempdata = data[:,features]
    #print(current_set_of_features, "\n")
    
    for i in range(0, data.shape[0]):
        object_to_classify = tempdata[i,1:]
        label_object_to_classify = tempdata[i,0]
        
        nearest_neighbor_distance = np.inf
        nearest_neighbor_location = np.inf
        
        for k in range(0, tempdata.shape[0]):
            if k != i:
                #for y in range(1, data.shape[1]):
                #    if ((y not in current_set_of_features) and (y != feature_to_add)):
                #        tempdata[k,y] = 0
                #print("Asking if " + str(i+1) + " is nearest neighbour with " + str(k+1))
                temp = np.subtract(object_to_classify, tempdata[k,1:])
                distance = math.sqrt(np.dot(temp, temp))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = tempdata[nearest_neighbor_location,0]
                
            
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified = number_correctly_classified + 1
        #print(nearest_neighbor_distance, "\n")
    return number_correctly_classified / data.shape[0]

def leave_one_out_cross_validation_backwards(data, current_set_of_features, feature_to_remove):
    number_correctly_classified = 0
    
    features = current_set_of_features
    features = np.insert(features, 0, 0, 0)
    features = np.delete(features, np.where(features == feature_to_remove))
    tempdata = data[:,features]
    
    for i in range(0, data.shape[0]):
        object_to_classify = tempdata[i,1:]
        label_object_to_classify = tempdata[i,0]
        
        nearest_neighbor_distance = np.inf
        nearest_neighbor_location = np.inf
        
        for k in range(0, tempdata.shape[0]):
            if k != i:
                temp = np.subtract(object_to_classify, tempdata[k,1:])
                distance = math.sqrt(np.dot(temp, temp))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = tempdata[nearest_neighbor_location,0]
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified = number_correctly_classified + 1
    return number_correctly_classified / data.shape[0]

def forward_select(data):
    starttime = time.time()
    current_set_of_features = [] #Initialize empty set
    
    best_acc_so_far = 0

    for i in range(0, data.shape[1]-1):
        print("On level "+str(i+1)+" of the search tree")
        feature_to_add_at_this_level = None
        
        for k in range(1, data.shape[1]):
            if (k in current_set_of_features):
                continue
            else:
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)
                print("    Using feature(s) ", current_set_of_features, " and considering addding feature ", k, " Accuracy is ", round(accuracy*100,2),"%\n")
                if accuracy > best_acc_so_far:
                    best_acc_so_far = accuracy
                    feature_to_add_at_this_level = k

        if feature_to_add_at_this_level:
            current_set_of_features.append(feature_to_add_at_this_level)
        print("On level "+str(i+1)+" I added feature "+str(feature_to_add_at_this_level)+" to current set.\n")
        print("Best feature set was ", current_set_of_features, ", accuracy is ", round(best_acc_so_far*100,2),"%\n")
    endtime = time.time()
    totaltime = round(endtime - starttime,2)
    print("Finished search! The best feature set is ",current_set_of_features)
    print("It has an accuracy of ", round(best_acc_so_far*100,2), "% And took ", totaltime, "seconds.")

def backward_elim(data):
    starttime = time.time()
    current_set_of_features = np.arange(1,data.shape[1]) #Initialize full set
    
    best_acc_so_far = 0

    for i in range(0, data.shape[1]-1):
        print("On level "+str(i+1)+" of the search tree")
        feature_to_remove_at_this_level = None
        
        for k in range(1, data.shape[1]):
            accuracy = leave_one_out_cross_validation_backwards(data, current_set_of_features, k)
            print("    Using feature(s) ", current_set_of_features, " and considering removing feature ", k, " Accuracy is ", round(accuracy*100,2),"%\n")
            if accuracy > best_acc_so_far:
                best_acc_so_far = accuracy
                feature_to_remove_at_this_level = k

        if feature_to_remove_at_this_level:
            current_set_of_features = np.delete(current_set_of_features, np.where(current_set_of_features == feature_to_remove_at_this_level))
        print("On level ",str(i+1)," I removed feature ",str(feature_to_remove_at_this_level)," from the current set")
    endtime = time.time()
    totaltime = round(endtime - starttime,2)
    print("Finished search! The best feature set is ",current_set_of_features)
    print(" It has an accuracy of ", round(best_acc_so_far*100,2), "% And took ", totaltime, "seconds.")

if __name__ == "__main__":
    main()