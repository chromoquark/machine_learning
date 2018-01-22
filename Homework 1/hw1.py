from math import log
from numpy import std
#
class attributeCounter():
	#This class counts the number of times each value is used for an attribute.
	#It also keeps track of how many of those were positives.  This allows us
	#to quickly find the percentage of positive examples from an attribute,
	#which is helpful to find the entropy.
	def __init__(self):
		#Initialize the dicts which we use to count.
		self.vals={}
		self.positive={}
		#Set the iterator index to -1, so that we can add 1 on the first
		#iteration and get the 0 index
		self.index=-1

	def __getitem__(self,a):
		#If the key is in vals, get it
		if a in self.vals:
			return self.vals[a]
		#Otherwise, return 0
		else:
			return 0

	def __setitem__(self,a,b):
		#Access the vals with the setter
		self.vals[a]=b

	def increment(self,a,pos):
		#Increment the vals counter
		self[a]+=1
		#If the positive value hasn't been encountered before, initialize it
		if a not in self.positive:
			self.positive[a]=0
		#If the example is positive for the label,increment the positive counter
		if pos:
			self.positive[a]+=1

	def __len__(self):
		#Return the number of values
		return len(self.vals)

	def __iter__(self):
		#Iterations return self to get the next iteree
		self.index=-1
		return self

	def __next__(self):
		#Looks for the next iteree.  Add one to the index
		self.index +=1
		#If we're at the end, start over
		if self.index == len(self):
			self.index = -1
			raise StopIteration
		#Return the next item, not the index
		return [a for a in self.vals][self.index]

	def informationGain(self):
		#Get the total numbe of examples used
		n=0
		for a in self.vals:
			n+=self.vals[a]
		#Find the total sum used in the information game
		sum=0
		#For each value
		for a in self.vals:
			#Get the probability of a positive label
			p=self.positive[a]/self.vals[a]
			#If p is 0 or 1, all examples have the same label, so entropy is 0
			if p==1 or p==0:
				continue
			#Add entropies
			sum-=(p*log(p,2)+(1-p)*log(1-p,2))*self.vals[a]/n
		#Subtract from 1 for the information gain
		return 1-sum

#may be optimizable
	def mostCommonLabel(self,label,notLabel):
		#Initialize the total and positive example sums
		sumPos=0
		sumTot=0
		#For each example, add to the sums
		for a in self.positive:
			sumPos+=self.positive[a]
			sumTot+=self.vals[a]
		#If more than half are positive, return the positive label
		if sumPos>sumTot/2:
			return label
		#Otherwise, return the negative one
		else:
			return notLabel

	
class example():
	#This stores a single training or test example.  It takes a feature vector
	#as input, with the label as the last feature
	def __init__(self,features):
		#Store features
		self.features=features
		#Store label
		self.label=features[-1]
		#Remove the label from the features
		del self.features[-1]

	def __getitem__(self,a):
		#Return the  value of feature a.
		return self.features[a]


class treeNode():
	#This implements the actual search tree.  The node has a childrn dict.
	#The keys to the dict are the specific values of the attribute, and the
	#node represents a decision to be made in the tree.
	#Children can be nodes, or char.  If it's char, it represents the label
	def __init__(self,attribute,possibleValues):
		#Initialize and store
		self.children={}
		self.attribute=attribute
		self.possibleValues=possibleValues
		self.possibleValues.append("Default")

	def addChild(self,child,val):
		#Add a child
		self.children[val]=child

	def __getitem__(self,i):
		#If the child exissts, return it
		if i in self.children:
			return self.children[i]
		#Otherwise, return the "Default" node value
		return self.children["Default"]


class id3Tree():
	#This object will read files, store the data, and then allows you to compute the
	#tree.  It also has functions for getting the max depth of the tree, computing
	#the tree's accuracy on the trainig set, and testing other data sets.
	def  __init__(self,fileName,Label,notLabel,depthLimit = float("inf"),handleQuestionMark=3):
		#The infinite depthLimit by default means there is no depth limit.
		#handleQuestionMark default of 3 indicates that there is no change, and that
		#the "?" will be treated as its own attribute value
		#If we have a single filename, change it to a list
		if not isinstance(fileName,list):
			fileName = [fileName]
		#Store the file names
		self.fileName = fileName
		#Initialize the list of examples and the counts of their attributes
		self.examples = []
		self.attributePossibleValues = []
		#Indicate that attributeCounters haven't been created yet.  This happens
		#only on the first iteration.
		self.firstIter = True
		#Store the labels and depth Limit
		self.Label = Label
		self.notLabel = notLabel
		self.depthLimit = depthLimit

		#Read all the files, and store their data
		for f in self.fileName:
			self.importData(f)
		
		#Choose what to replace "?" with in the data
		self.handleMissing(handleQuestionMark)


		#Compute the tree, using the set of all training examples, and 
		self.id3Tree = self.id3(self.examples,[i for i,a in enumerate(self.attributePossibleValues)])

	def importData(self,fileName):
		#Open the file to read
		f = open(fileName,'r')
		#For every line
		for line in f:
			#Make a training example out of a feature vector (formed by separating the line by
			#the "\t" and "," delimiters)
			tE = example(line.replace("\n","").replace("\t",",").split(","))
			self.examples.append(tE)
			#If this is the first iteration, then create the attribute counters
			if self.firstIter:
				for i in tE.features:
					self.attributePossibleValues.append(attributeCounter())
				#This is no longer the first iteration
				self.firstIter = False
			#For every attribute, increment the attribute counters
			for i, attribute in enumerate(tE.features):
				self.attributePossibleValues[i].increment(attribute,tE.label==self.Label)

	def handleMissing(self,handleQuestionMark):
		#How to handle "?" in the data
		#Option 1 is choose the most common attribute for that value
		if handleQuestionMark == 1:
			#For every attribute
			for i,aV in enumerate(self.attributePossibleValues):
				#If there is a "?"
				if "?" in aV:
#
					temp = aV.vals.copy()
					del temp["?"]
					#Find what the max value for that attribute is
					temp = [(value,key) for key,value in temp.items()]
					val = max(temp)[1]
					#In all the examples, change the attribute to the max attribute
					for t in self.examples:
						if t.features[i]=="?":
							t.features[i]=val
					#Merge the max value and "?" counts, then delete the "?" entry
					aV.vals[val]+=aV.vals["?"]
					aV.positive[val]+=aV.positive["?"]
					del aV.vals["?"]
					del aV.positive["?"]
		#Option 2 is choose the most common attribute for that value from all samples
		#with the same label
		elif handleQuestionMark == 2:
			#For every attribute
			for i,aV in enumerate(self.attributePossibleValues):
				#If there is a "?"
				if "?" in aV:
#
					temp = aV.vals.copy()
					del temp["?"]
					temp = [(value,key) for key,value in temp.items()]
					#Find the most common positive attribute value
					valPos = max(temp)[1]

					temp = aV.vals.copy()
					for a in temp:
						temp[a]-=aV.positive[a]
					del temp["?"]
					temp = [(value,key) for key,value in temp.items()]
					#Find the most common negative attribute value
					valNeg = max(temp)[1]
					#For each example, if there is a "?", we select a value.
					for t in self.examples:
						if t.features[i]=="?":
							#If the label is positive, then use the most common positive labels
							if t.label==self.Label:
								val=valPos
								#Update the positive count
								aV.increment(val,True)
							#If the label is negative, then use the most common negative labels
							else:
								val=valNeg
								aV.increment(val,False)
						#Update the example
						t.features[i]=val
						#Update the counter
					#After looking at all training examples, delete the "?" entries in the counters
					del aV.vals["?"]
					del aV.positive["?"]

	def id3(self,S,attr):
		#Make a list of all examples that do not match the label 
		label = S[0].label
		labelsNotSame = [t for t in S if t.label!=label]
		#If labelsNotSame is not empty, then there is a different label.
		#If they're all the same, labelsNotSame==[], so return the label.
		if labelsNotSame==[]:
			return label
		#If not all the labels are the same
		#Compute all the information gains

		#Get an attribute counter for this subset
		attributePossibleValues=[]
		for i in attr:
			attributePossibleValues.append(attributeCounter())
		for e in S:
			for i,a in enumerate(attr):
				attributePossibleValues[i].increment(e.features[a],e.label==self.Label)

		#Get the information gains for this subset, and then get the attribute that is has the most
		#information gain.  Also obtain its index in this subset of attributes that have not been
		#split upon yet.
		informationGains = [attributePossibleValues[i].informationGain() for i,a in enumerate(attr)]
		bestAttributeSubIndex = informationGains.index(max(informationGains))
		bestAttribute = attr[bestAttributeSubIndex]

		#Create a treeNode. This will be the root if this is the first function call, but
		#the id3 algorithm is recursive, so it will simply return this node to its parent
		root = treeNode(bestAttribute, [v for v in self.attributePossibleValues[bestAttribute]])
		#For all possible values (decisions) to take for this node
		for v in root.possibleValues:
			#Get the set of the examples that have that attribute
			Sv = [example for example in S if example[bestAttribute]==v]
			#See fi we've reached the depth limit
			reachedDepthLimit = len(self.attributePossibleValues)-len(attr)+1==self.depthLimit
			#If there are no values of v in the examples
			if Sv==[]:
#				#Count all the examples in the whole set S
#				labelCounter = attributeCounter()
#				for t in S:
#					labelCounter.increment(t.label,t.label==self.Label)
#				#Make the most common label
#				root.addChild(labelCounter.mostCommonLabel(label,self.notLabel),v)
				root.addChild(attributePossibleValues[bestAttributeSubIndex].mostCommonLabel(label,self.notLabel),v)
			#If we've reached the depth limit
			elif reachedDepthLimit:
				#Count all the examples in Sv
				labelCounter = attributeCounter()
				for t in Sv:
					labelCounter.increment(t.label,t.label==self.Label)
				#Make the most common label in the child
				root.addChild(labelCounter.mostCommonLabel(label,self.notLabel),v)
			#If we need a new decision
			else:
				#Add a childenode recursively using the attr without the current decision attribute
				attrSub = attr.copy()
				del attrSub[bestAttributeSubIndex]
				root.addChild(self.id3(Sv,attrSub),v)
		#Add a default option.  This is so that if an attribute value which is not in the training set is encountered
		#during testing, it has a value to choose.  
		root.addChild(attributePossibleValues[bestAttributeSubIndex].mostCommonLabel(self.Label,self.notLabel),"Default")
		#Return the newly formed node
		return root

	def query(self,features,node = None):
		#Find the label for a set of features
		#If there's no node to explore, look at the root
		if node == None:
			node=self.id3Tree
		#If the node is a tree node, then get it's value '
		if isinstance(node,treeNode):
			val = features[node.attribute]
			#Recursively query what the label is
			return self.query(features,node[val])
		#Return the label
		return node

	def maxDepth(self,node=None):
		#Return the max depth of the tree
		#If we haven't looked at a node yet, get the root
		if node==None:
			node=self.id3Tree
		#Get all the chilrden that are decision nodes for our node
		children = [node.children[c] for c in node.children if isinstance(node.children[c],treeNode)]
		#If there are no decision-children, return 0 (for a label not counting in depth)
		#or 1 (if the label does count)
		if children == []:
			return 0
		#Otherwise, gt all the depths ofthe children (recursively)
		childrenDepths = [self.maxDepth(c) for c in children]
		#Return the maximum child depth, incremented by 1
		return max(childrenDepths)+1

	def testTreeImplementation(self):
		#Tests the tree's accuracy against the training set
		#Initialize the number of correct
		correct=0
		#For each example
		for q in self.examples:
			#Query the example
			s = self.query(q.features)
			#count it correct if the label is correct
			if s==q.label:
				correct+=1
		#Print the result.
		print("Training Data Error = "+"%2.1d"%100*(correct/len(self.examples)))

	def evaluateTestSet(self,file):
		#Return the decimal percentage of examples correctly classified in a test set
		#Open the file
		f = open(file,'r')
		#Initialize the positive and total counts
		count=0
		correct=0
		#For each example
		for line in f:
			#Form an example from the line
			testExample = example(line.replace("\n","").replace("\t",",").split(","))
			#Increment the count
			count+=1
			#If it queries to the correct label, increment the correct count
			if self.query(testExample.features)==testExample.label:
				correct+=1
		#Return the accuracy
		return correct/count

#If this is being run as a script, and not imported as a library, then do the following
if __name__=="__main__":
	def crossValidation(folder,depths,qMHandler=3):
		#Define a cross validation using the missing data handler (qMHandler), and the max depths (depths)
		#Initialize the storage for the errors
		depthPerformances=[]
		#Get all the files in the specified folder.  These files are a segmented training set for cross validation
		files=[]
		for i in range(0,6):
			files.append("data/"+folder+"/CVSplits/training_0"+"%d"%i+".data")
		#For each depth
		for d in depths:
			#Initialize the performance array, which will store the tests of the same depth
			p=[]
			#For all files
			for i,f in enumerate(files):
				#Remove that file from the list
				fileSub=files.copy()
				del fileSub[i]
				#Make a decision tree out of the other files
				decisionTree = id3Tree(fileSub,"e","p",depthLimit=d,handleQuestionMark=qMHandler)
				#Save the performance of the current file
				p.append(decisionTree.evaluateTestSet(f))
			#Put the performance array in the total performance array
			depthPerformances.append(p)
		#Initialize mean and standard deviation arrays
		meanAcc = []
		stdDev = []
		#For each entry in depth performance
		for p in depthPerformances:
			#Initialize the sum to 0, then get the sum
			sum=0
			for i in p:
				sum+=i
			#Append the mean and standard deviations to their lists.  Note that we add them
			#as percents, on a 0-100 scale, not as a decimal 0-1.
			meanAcc.append(100*(sum/len(p))),
			stdDev.append(100*std(p))
		#Print the means and standard deviations
		print("Mean Accuracy ",meanAcc)
		print("Standard Deviation ",stdDev)
		#Return the depth with the smallest index, as well as its error
		return depths[meanAcc.index(max(meanAcc))], max(meanAcc)

	def crossValidationQMHandler(folder,qMHandler=[1,2,3]):
		#Define a cross validation using the missing data handlers (qMHandler).  Assume no depthlimit
		#Initialize the storage for the errors
		methodPerformances=[]
		#Get all the files in the specified folder.  These files are a segmented training set for cross validation
		files=[]
		for i in range(0,6):
			files.append("data/"+folder+"/CVSplits/training_0"+"%d"%i+".data")
		#For each missing data handler
		for d in qMHandler:
			#Initialize the performance array, which will store the tests of the same depth
			p=[]
			#For all files
			for i,f in enumerate(files):
				#Remove that file from the list
				fileSub=files.copy()
				del fileSub[i]
				#Make a decision tree out of the other files
				decisionTree = id3Tree(fileSub,"e","p",handleQuestionMark=d)
				#Save the performance of the current file
				p.append(decisionTree.evaluateTestSet(f))
			#Put the performance array in the total performance array
			methodPerformances.append(p)
		#Initialize mean and standard deviation arrays
		meanAcc = []
		stdDev = []
		#For each entry in depth performance
		for p in methodPerformances:
			#Initialize the sum to 0, then get the sum
			sum=0
			for i in p:
				sum+=i
			#Append the mean and standard deviations to their lists.  Note that we add them
			#as percents, on a 0-100 scale, not as a decimal 0-1.
			meanAcc.append(100*(sum/len(p))),
			stdDev.append(100*std(p))
		#Print the means and standard deviations
		print("Mean Accuracy ",meanAcc)
		print("Standard Deviation ",stdDev)
		#Return the depth with the smallest index, as well as its error
		return qMHandler[meanAcc.index(max(meanAcc))], max(meanAcc)


	print("Part A")
	#Train on the part A data, and print the accuracies on the training set and the test set, and print the max depth 
	decisionTree = id3Tree("./data/SettingA/training.data","e","p")
	print("Training Set A Accuracy =",100*(decisionTree.evaluateTestSet("./data/SettingA/training.data")))
	print("Test Set A Accuracy =",100*(decisionTree.evaluateTestSet("./data/SettingA/test.data")))
	print("Max Depth=",decisionTree.maxDepth())

	#Perform cross-validation for the A data, and print the results
	limit = crossValidation("SettingA",[1,2,3,4,5,10,15,20])
	print("Depth Limit=",limit)
	decisionTree = id3Tree("./data/SettingA/training.data","e","p",depthLimit=limit[0])
	print("Depth-Limited Training Set A Accuracy =",100*(decisionTree.evaluateTestSet("./data/SettingA/training.data")))
	print("Depth-Limited Test Set A Accuracy =",100*(decisionTree.evaluateTestSet("./data/SettingA/test.data")))



	print("\nPart B")
	#Train on the part B data, and print the accuracies on the training set and the test set for A and B, and print the max depth 
	decisionTree = id3Tree("./data/SettingB/training.data","e","p")
	print("Training Set B Accuracy =",100*(decisionTree.evaluateTestSet("./data/SettingB/training.data")))
	print("Test Set B Accuracy =",100*(decisionTree.evaluateTestSet("./data/SettingB/test.data")))
	print("Training Set A Accuracy =",100*(decisionTree.evaluateTestSet("./data/SettingA/training.data")))
	print("Test Set A Accuracy =",100*(decisionTree.evaluateTestSet("./data/SettingA/test.data")))
	print("Max Depth=",decisionTree.maxDepth())

	#Perform cross-validation for the B data, and print the results
	limit = crossValidation("SettingB",[1,2,3,4,5,10,15,20])
	print("Depth Limit=",limit)
	decisionTree = id3Tree("./data/SettingB/training.data","e","p",depthLimit=limit[0])
	print("Depth-Limited Training Set B Accuracy =",100*(decisionTree.evaluateTestSet("./data/SettingB/training.data")))
	print("Depth-Limited Test Set B Accuracy =",100*(decisionTree.evaluateTestSet("./data/SettingB/test.data")))


	print("\nPart C")
	#Perform cross-validation for the missing data handler methods
	limit, err = crossValidationQMHandler("SettingC")
	#Form the decision tree with the best missing data handler, and then print its accuracy on the test set
	decisionTree = id3Tree("./data/SettingC/training.data","e","p",handleQuestionMark=3)
	print("Test Set",100*(decisionTree.evaluateTestSet("./data/SettingC/test.data")))


	#Pokemon data set as debugger
	#decisionTree = id3Tree("pokemon.data","y",)
	#decisionTree.testTreeImplementation()
