import math
from random import randint

class Node:
    
    def __init__(self,data,train):
        self.data = data
        self.train = train
        self.left = None
        self.right = None
        


def createPostPrunedTree(data,attributes,root,L,K):
    postPrunedroot = root
    L = int(L)
    for i in range(1,L+1):
        temp = copy(root)
        M = randint(1,int(K))
        for j in range(1,M+1):
            N = countNonLeafNodes(temp)
            if N > 0:
                P = randint(1,int(N))
                node = findPNode(temp,P)
                node.data = node.train
                node.left = None
                node.right = None
        bestTreeAccuracy = accuracy(postPrunedroot,data,attributes)
        tempTreeAccuracy = accuracy(temp,data,attributes)
        
        if tempTreeAccuracy > bestTreeAccuracy:
            postPrunedroot = temp
    return postPrunedroot


def countNonLeafNodes(root):
    if root == None or (root.left == None and root.right == None):
        return 0
    left = countNonLeafNodes(root.left)
    right = countNonLeafNodes(root.right)
    return 1 + left + right
    
def findPNode(root,P):
    if root == None:
        return None
    if P == 1:
        return root
    q1 = []
    q2 = []
    q1.append(root)
    count = P
    
    while(len(q1) > 0 or len(q2) > 0):
        
        while(len(q1) > 0):
            popped_node = q1.pop()
            if not(popped_node.left == None and popped_node.right == None):
                count = count-1
            if count == 0:
                return popped_node
            if popped_node.left:
                q2.append(popped_node.left)
            if popped_node.right:
                q2.append(popped_node.right)
        while(len(q2) > 0):
            popped_node = q2.pop()
            if not(popped_node.left == None and popped_node.right == None):
                count = count-1
            if count == 0:
                return popped_node
            if popped_node.left:
                q1.append(popped_node.left)
            if popped_node.right:
                q1.append(popped_node.right)

def accuracy(root,data,attributes):
    count = 0
    
    for i in range(0,len(data)):
        node = root
        while(True):
            if (node.data == '1' or node.data == '0'):
                if node.data == data[i][attributes.index("Class")]:
                    count = count + 1
                break    
            else:
                index = attributes.index(node.data)
                if data[i][index] == str(0):
                    node = node.left
                else:
                    node = node.right                
    return float(count/len(data))


def copy(root):
    node = root
    createdNode = None
    if node == None:
        return None
    if node.left == None and root.right == None:
        data = node.data
        default = node.train
        createdNode = Node(data,default)
    else:
        data = node.data
        default = node.train
        createdNode = Node(data,default)
        createdNode.left = copy(node.left)
        createdNode.right = copy(node.right)
    return createdNode



def createTreeWithInfoGain(data, attributes, target):
    default = highestCountedClass(attributes, data, target)
    values = [record[attributes.index(target)] for record in data]
    data = data[:]
    if default == "":
        default = str(0)
    
    if not data or (len(attributes) - 1) <= 0:
        return Node(default,default)
    
    elif values.count(values[0]) == len(values):
        return Node(values[0],values[0])
    else:
        
        best = chooseBestAttributeWithInfoGain(data, attributes, target)
        root = Node(best,default)
        
        for val in range(0,2):
            
            examples = getSubsetWithBestAttribute(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            node = createTreeWithInfoGain(examples, newAttr, target)
            
            
            if val == 0:
                root.left = node
            else:
                root.right = node
    
    return root



def createTreeWithVarianceImpurity(data, attributes, target):
    default = highestCountedClass(attributes, data, target)
    if default == "":
        default = 0
    values = [record[attributes.index(target)] for record in data]
    data = data[:]
    
    if not data or (len(attributes) - 1) <= 0:
        return Node(default,default)
    
    elif values.count(values[0]) == len(values):
        return Node(values[0],values[0])
    else:
        
        best = chooseBestAttributeWithVarianceImpurity(data, attributes, target)
        root = Node(best,default)
        
        for val in range(0,2):
            
            examples = getSubsetWithBestAttribute(data, attributes, best, val)
            changedAttributes = attributes[:]
            changedAttributes.remove(best)
            node = createTreeWithVarianceImpurity(examples, changedAttributes, target)
            
            
            if val == 0:
                root.left = node
            else:
                root.right = node
    
    return root




def varianceImpurity(attributes,data,targetAttr):
    
    freqencies = {}
    entropy = 1.0 
    i = attributes.index(targetAttr) 
    for entry in data:
        if (entry[i] in freqencies):
            freqencies[entry[i]] += 1.0
        else:
            freqencies[entry[i]] = 1.0
    if (len(freqencies) == 1):
        return 0
    for freqency in freqencies.values():
        entropy *= (freqency/len(data))
    return entropy

def entropy(attributes, data, targetAttribute):
    valueFreqencies = {}
    entropy = 0.0 
    i = attributes.index(targetAttribute) 
    for entry in data:
        if (entry[i] in valueFreqencies):
            valueFreqencies[entry[i]] += 1.0
        else:
            valueFreqencies[entry[i]]  = 1.0
 
    for freq in valueFreqencies.values():
        entropy += (-freq/len(data)) * math.log(freq/len(data), 2)     
    return entropy

def gainWithInfoGain(attributes, data, attribute, targetAttr):   
    valueFreqencies = {}
    subsetEntropy = 0.0 
    i = attributes.index(attribute)    
    for entry in data:
        if (entry[i] in valueFreqencies):
            valueFreqencies[entry[i]] += 1.0
        else:
            valueFreqencies[entry[i]]  = 1.0    
    for value in valueFreqencies.keys():
        valueProportion        = valueFreqencies[value] / sum(valueFreqencies.values())
        dataSubset     = [entry for entry in data if entry[i] == value]
        subsetEntropy += valueProportion * entropy(attributes, dataSubset, targetAttr)
    
    return (entropy(attributes, data, targetAttr) - subsetEntropy)


def gainWithVarianceImpurity(attributes, data, attribute, targetAttr):
   
    valueFreqencies = {}
    subsetEntropy = 0.0
    i = attributes.index(attribute)
    for entry in data:
        if (entry[i] in valueFreqencies):
            valueFreqencies[entry[i]] += 1.0
        else:
            valueFreqencies[entry[i]]  = 1.0
    
    for val in valueFreqencies.keys():
        valueProportion        = valueFreqencies[val] / sum(valueFreqencies.values())
        dataSubset     = [entry for entry in data if entry[i] == val]
        subsetEntropy += valueProportion * varianceImpurity(attributes, dataSubset, targetAttr)
    
    return (varianceImpurity(attributes, data, targetAttr) - subsetEntropy)



def chooseBestAttributeWithVarianceImpurity(data, attributes, target):
    bestAttribute = attributes[0]
    maximumGain = 0;
    for attribute in attributes:
        if attribute == target:
            continue
        newGain = gainWithVarianceImpurity(attributes, data, attribute, target)
        if newGain>maximumGain:
            maximumGain = newGain
            bestAttribute = attribute
    return bestAttribute


def chooseBestAttributeWithInfoGain(data, attributes, target):
    bestAttribute = attributes[0]
    maximumGain = 0;
    for attribute in attributes:
        if attribute == target:
            continue
        newGain = gainWithInfoGain(attributes, data, attribute, target)
        if newGain>maximumGain:
            maximumGain = newGain
            bestAttribute = attribute
    return bestAttribute



def getSubsetWithBestAttribute(data, attributes, best, val):
    subset = [[]]
    index = attributes.index(best)
    for tuple in data:
        
        if (int(tuple[index]) == val):
            entry = []
            
            for i in range(0,len(tuple)):
                if(i != index):
                    entry.append(tuple[i])
            subset.append(entry)
    subset.remove([])
    return subset



def highestCountedClass(attributes, data, target):
    
    index = attributes.index(target)
    freqencies = {}
    
    for entry in data:
        if (entry[index] in freqencies):
            freqencies[entry[index]] += 1 
        else:
            freqencies[entry[index]] = 1
    max = 0
    highestCountedClass = ""
    for key in freqencies.keys():
        if freqencies[key]>max:
            max = freqencies[key]
            highestCountedClass = key
    return highestCountedClass