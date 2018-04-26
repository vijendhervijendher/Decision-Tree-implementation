import DecisionTree
import sys
import os

def main():
    
    
    toPrint = sys.argv[6]
    
    toOpenFile = open(os.path.dirname(os.path.realpath(__file__))  + sys.argv[3])
    
    data = [[]]
    for line in toOpenFile:
        line = line.strip("\r\n")
        data.append(line.split(','))
    data.remove([])
    attributes = data[0]
    target = attributes[len(attributes)-1]
    data.remove(attributes)
    
    
    toOpenFile = open(os.path.dirname(os.path.realpath(__file__)) +  sys.argv[4])
   
    validationData = [[]]
    for line in toOpenFile:
        line = line.strip("\r\n")
        validationData.append(line.split(','))
    validationData.remove([])
    validationAttributes = validationData[0]
    validationData.remove(validationAttributes)
    

    
    toOpenFile = open(os.path.dirname(os.path.realpath(__file__)) +  sys.argv[5])
   
    testData = [[]]
    for line in toOpenFile:
        line = line.strip("\r\n")
        testData.append(line.split(','))
    testData.remove([])
    testAttributes = testData[0]
    testData.remove(testAttributes)
    
    root = DecisionTree.createTreeWithInfoGain(data, attributes, target)
    if toPrint == 'yes':
        print ("Generated Tree with Information Gain Heuristic : ")
        printTree(root,0)
    accuracy = DecisionTree.accuracy(root,testData,attributes)
    print("\n" + "Accuracy of tree with Information Gain Heuristic : ")
    print("\n" + str(accuracy))
    
    root2 = DecisionTree.createTreeWithVarianceImpurity(data,attributes,target)
    if toPrint == 'yes': 
        print ("Generated Tree with Variance Impurity Heuristic : ")
        printTree(root2,0)
    accuracy = DecisionTree.accuracy(root,testData,attributes)
    print("\n" + "Accuracy of tree with Variance Impurity Heuristic : ")
    print("\n" + str(accuracy))
    
    L = sys.argv[1]
    K = sys.argv[2]
    postPrunedRoot = DecisionTree.createPostPrunedTree(validationData,validationAttributes,root,L,K)
    postPrunedAccuracy = DecisionTree.accuracy(postPrunedRoot,testData,testAttributes)
    print("\n"+ "accuracy of pruned tree with L =  " + str(L) + " and K = " + str(K) + "  is " + str(postPrunedAccuracy))
    
def printTree(root,count):
        
        if root.left == None and root.right == None:
           print(root.data,end="",flush=True)
        else:
            print ("\n")
            for i in range(0,count):
                print("\t",end = "",flush=True)
            print (root.data + " = 0 : ",end="",flush=True)
            printTree(root.left,count+1)
            print ("\n")
            for i in range(0,count):
                print("\t",end = "",flush=True)
            print (root.data + " = 1 : ",end="",flush=True)
            printTree(root.right,count+1)
	

    
if __name__ == '__main__':
    main()

