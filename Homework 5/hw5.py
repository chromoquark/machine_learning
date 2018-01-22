import numpy as np
import random












import math

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
			sum-=(p*math.log(p,2)+(1-p)*math.log(1-p,2))*self.vals[a]/n
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
	def __init__(self):
		self.Label=1
		self.notLabel=-1

	def id3(self,S,attr):
		#Make a list of all examples that do not match the label 
		label = S[0].lab
		labelsNotSame = [t for t in S if t.lab!=label]
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
				attributePossibleValues[i].increment(e.attr[a],e.lab==self.Label)

		#Get the information gains for this subset, and then get the attribute that is has the most
		#information gain.  Also obtain its index in this subset of attributes that have not been
		#split upon yet.
#		print("APV",len(attributePossibleValues),len(attr))
		informationGains = [attributePossibleValues[i].informationGain() for i,a in enumerate(attr)]
#		print("IG",len(informationGains))
		bestAttributeSubIndex = informationGains.index(max(informationGains))
		bestAttribute = attr[bestAttributeSubIndex]

		#Create a treeNode. This will be the root if this is the first function call, but
		#the id3 algorithm is recursive, so it will simply return this node to its parent
		root = treeNode(bestAttribute, [v for v in attributePossibleValues[bestAttributeSubIndex]])
		#For all possible values (decisions) to take for this node
		for v in root.possibleValues:
			#Get the set of the examples that have that attribute
			Sv = [example for example in S if example[bestAttribute]==v]
#			print("Sv",len(Sv))
			#See fi we've reached the depth limit
#			reachedDepthLimit = len(self.attributePossibleValues)-len(attr)+1==self.depthLimit
			#If there are no values of v in the examples
			if Sv==[] or len(attr)==1:
#				print(1)
				root.addChild(attributePossibleValues[bestAttributeSubIndex].mostCommonLabel(label,self.notLabel),v)
			#If we need a new decision
			else:
#				print(2)
				#Add a childenode recursively using the attr without the current decision attribute
				attrSub = attr.copy()
#				print("AttrSub",len(attrSub))
				del attrSub[bestAttributeSubIndex]
#				print("AttrSub",len(attrSub))
				root.addChild(self.id3(Sv,attrSub),v)
		#Add a default option.  This is so that if an attribute value which is not in the training set is encountered
		#during testing, it has a value to choose.  
		root.addChild(attributePossibleValues[bestAttributeSubIndex].mostCommonLabel(self.Label,self.notLabel),"Default")
		#Return the newly formed node
		self.root=root
		return root

	def query(self,features,node = None):
		#Find the label for a set of features
		#If there's no node to explore, look at the root
		if node == None:
			node=self.root
		#If the node is a tree node, then get it's value '
		if isinstance(node,treeNode):
			val = features[node.attribute]
			#Recursively query what the label is
			return self.query(features,node[val])
		#Return the label
		return node

	def testTreeImplementation(self):
		#Tests the tree's accuracy against the training set
		#Initialize the number of correct
		correct=0
		#For each example
		for q in self.test:
			#Query the example
			s = self.query(q.attr)
			#count it correct if the label is correct
			if s==q.lab:
				correct+=1
		#Print the result.
#		print("Training Data Error = "+"%2.1d"%100*(correct/len(self.examples)))











class example():
	def __init__(self,attr,lab):
		self.attr=np.array(attr)
		self.lab=lab
	def __getitem__(self,a):
		#Return the  value of feature a.
		return self.attr[a]


class data():
	def __init__(self,folder,crossValNum=5):
		self.crossValNum=crossValNum
		self.examples=[]
		self.test=[]

		labelsFile = open(folder+"train.labels","r")
		for i,f in enumerate(open(folder+"train.data","r")):
			lab = float(labelsFile.readline().replace("\n",""))
			f = f.replace("\n","").split(" ")
			if f[-1]=="": del f[-1]
			for i,ff in enumerate(f): f[i]=float(f[i])
			self.examples.append(example([1]+f,lab))

		self.learnOn=self.examples

		labelsFile = open(folder+"test.labels","r")
		for f in open(folder+"test.data","r"):
			lab = float(labelsFile.readline().replace("\n",""))
			f = f.replace("\n","").split(" ")
			if f[-1]=="": del f[-1]
			for i,ff in enumerate(f): f[i]=float(f[i])
			self.test.append(example([1]+f,lab))

		self.testOn=self.test

		self.w=np.zeros(len(self.learnOn[0].attr))
		self.crossVal=self.makeCrossVal(crossValNum,self.examples)

	def makeCrossVal(self,crossValNum,set):
		crossVal=[]
		for i in range(0,crossValNum):
			start = int( i*len(set)/crossValNum )
			stop = int( (i+1)*len(set)/crossValNum )
			crossVal.append(set[start:stop])
		return crossVal

	def gamma(self,t):
		return self.gamma0/(1+self.gamma0*t/self.C)

	def learn(self,epochs=3,gamma0=0.01,C=1):
		self.w=np.zeros(len(self.learnOn[0].attr))
		self.gamma0 = gamma0
		self.C = C
		random.seed(50)
#		t=0
#		self.wOld=deepcopy(self.w)
#		while np.linalg.norm(self.wOld-self.w)>10**-1 or np.linalg.norm(self.w)==0:
#			print(np.linalg.norm(self.wOld-self.w))
#			self.wOld=deepcopy(self.w)
#			t+=1
		for t in range(0,epochs*len(self.learnOn)):
			random.shuffle(self.learnOn)
			e=random.sample(self.learnOn,1)[0]
			if e.lab*sum(self.w*e.attr)<=1:
				self.w=(1-self.gamma(t))*self.w+self.gamma(t)*C*e.lab*e.attr
			else:
				self.w=(1-self.gamma(t))*self.w
	def acc(self):
		count=0
		correct=0
		for e in self.learnOn:
			count+=1
			if sum(self.w*e.attr)*e.lab>=0:
				correct+=1
#		print("Training Accuracy:",correct,count)
		train=correct/count
		count=0
		correct=0
		for e in self.testOn:
			count+=1
			if sum(self.w*e.attr)*e.lab>=0:
				correct+=1
#		print("Training Accuracy:",correct,count)
		test=correct/count
		return train,test

	def f1Score(self):
		tp=0
		fp=0
		fn=0
		for e in self.learnOn:
			if sum(self.w*e.attr)>=0 and e.lab>=0:
				tp+=1
			elif sum(self.w*e.attr)<0 and e.lab>=0:
				fn+=1
			elif sum(self.w*e.attr)>=0 and e.lab<0:
				fp+=1
		pl=tp/(tp+fp)
		rl=tp/(tp+fn)
		f1l=2*pl*rl/(pl+rl)
		tp=0
		fp=0
		fn=0
		for e in self.testOn:
			if sum(self.w*e.attr)>=0 and e.lab>=0:
				tp+=1
			elif sum(self.w*e.attr)<1 and e.lab>=0:
				fn+=1
			elif sum(self.w*e.attr)>=0 and e.lab<0:
				fp+=1
		pt=tp/(tp+fp)
		rt=tp/(tp+fn)
		f1t=2*pt*rt/(pt+rt)
		return pl,rl,f1l,pt,rt,f1t

	def convertToTreeSpace(self,set,N):
		learnOn = []
		for E in set:
			attr=[1]
			for n in range(len(self.trees)):
				attr.append( self.trees[n].query(E.attr) )
			learnOn.append(example(attr,E.lab))
		return learnOn

	def treeCrossValStep(self,N,k,passes,c,g):
		perfSub=[]
		for i in range(0,self.crossValNum):
			learnOn = []
			for j in range(0,self.crossValNum):
				if i!=j: learnOn.append(self.crossVal[j])
			learnOn=sum(learnOn,[])
			
			#Make the decision Trees
			self.trees=[]
			m=int(len(self.examples))
			for n in range(N):
				S=[]
				for M in range(m):
					S.append(random.sample(learnOn,1)[0])
				numAttr = len(self.test[0].attr)
				attrsToUse=random.sample([i for i in range(numAttr)],k)
				root=id3Tree()
				root.id3(S,attrsToUse)
				self.trees.append(root)

			self.learnOn=self.convertToTreeSpace(learnOn,N)
			self.testOn=self.convertToTreeSpace(self.crossVal[i],N)
			self.learn(passes,gamma0=g,C=c)
			tr,te=self.acc()
			perfSub.append(te)
		return np.mean(perfSub)

	def prepTreeLearner(self,k=8,N=5,binSize=5,passes=1):
		random.seed(100)
		self.binSize=binSize
		for e in self.examples:
			for i in range(len(e.attr)):
				e.attr[i]=round(  e.attr[i]/binSize  )*binSize
		for e in self.test:
			for i in range(len(e.attr)):
				e.attr[i]=round(  e.attr[i]/binSize  )*binSize

		C = [2**i for i in range(-5,5)]
		G = [2**-i for i in range(0,10)]
		c=0.5
		g=0.03125
		for I in range(0,0):
			perf=[]
			for c in C:
				p=self.treeCrossValStep(N,k,passes,c,g)
				perf.append(p)
#				print(c,p)
#			print("C Values",N,perf,C)
			c = C[perf.index(max(perf))]
#			print("C",c)

			perf=[]
			for g in G:
				perf.append( self.treeCrossValStep(N,k,passes,c,g) )
#			print("gamma0 Values",N,perf,G)
			g = G[perf.index(max(perf))]
#			print("G",g)


		learnOn=self.examples
		
		#Make the decision Trees
		self.trees=[]
		m=int(len(self.examples))
		for n in range(N):
			S=[]
			for M in range(m):
				S.append(random.sample(learnOn,1)[0])
			numAttr = len(self.test[0].attr)
			attrsToUse=random.sample([i for i in range(numAttr)],k)
			root=id3Tree()
			root.id3(S,attrsToUse)
			self.trees.append(root)

		self.learnOn=self.convertToTreeSpace(self.examples,N)
		self.testOn=self.convertToTreeSpace(self.test,N)
		self.learn(passes,gamma0=g,C=c)
#		tr,te=self.acc()

		return c,g		


from copy import deepcopy

#Problem 3.1.1
hw = data("./data/handwriting/")
hw.learn()
train,test = hw.acc()
print("3.1.1")
print("Training Accuracy",train)
print("Testing Accuracy",test)


#Problem 3.1.2
print("3.1.2")
ma = data("./data/madelon/madelon_")
C = [3**i for i in range(-7,2)]
G = [3**-i for i in range(1,11)]
g=G[8]
for I in range(0,1):
	perf=[]
	for c in C:
#		print(c,C)
		for i in range(0,ma.crossValNum):
			perfSub=[]
			learnOn = []
			for j in range(0,ma.crossValNum):
				if i!=j: learnOn.append(ma.crossVal[j])
			ma.learnOn=sum(learnOn,[])
			ma.testOn=ma.crossVal[i]
			ma.learn(C=c,gamma0=g)
			tr,te=ma.acc()
			perfSub.append(te)
		perf.append(np.mean(perfSub))
	print("C Values",perf,C)
	c = C[perf.index(max(perf))]

	perf=[]
	for g in G:
#		print(g,G)
		for i in range(0,ma.crossValNum):
			perfSub=[]
			learnOn = []
			for j in range(0,ma.crossValNum):
				if i!=j: learnOn.append(ma.crossVal[j])
			ma.learnOn=sum(learnOn,[])
			ma.testOn=ma.crossVal[i]
			ma.learn(C=c,gamma0=g)
			tr,te=ma.acc()
			perfSub.append(te)
		perf.append(np.mean(perfSub))
	print("gamma0 Values",perf,G)
	g = G[perf.index(max(perf))]


ma.learnOn=ma.examples
ma.testOn=ma.test
ma.learn(C=c,gamma0=g)
train,test = ma.acc()
print("Training Accuracy",train)
print("Testing Accuracy",test)
print("C:",c)
print("gamma0:",g)


#Problem 3.1.3
print("3.1.3")
pl,rl,f1l,pt,rt,f1t=hw.f1Score()
print("For handwriting data training set")
print("Precision:",pl)
print("Recall:",rl)
print("F1:",f1l)
print("For handwriting data test set")
print("Precision:",pt)
print("Recall:",rt)
print("F1:",f1t)
pl,rl,f1l,pt,rt,f1t=ma.f1Score()
print("For madelon data training set")
print("Precision:",pl)
print("Recall:",rl)
print("F1:",f1l)
print("For madelon data test set")
print("Precision:",pt)
print("Recall:",rt)
print("F1:",f1t)

#Problem 3.2.1
print("3.2.1")
hw = data("./data/handwriting/")
hw.prepTreeLearner(binSize=1,k=8)
print(hw.acc())
print("Increased k from 8 to 100")
hw.prepTreeLearner(binSize=1,k=100)
print(hw.acc())

#Problem 3.2.2
print("3.2.2")
ma = data("./data/madelon/madelon_")
ma.finalTest=deepcopy(ma.test)
ma.finalExamples=deepcopy(ma.examples)
ma.examples=ma.finalExamples[0:int(.8*len(ma.finalExamples))]
ma.test=ma.finalExamples[int(.8*len(ma.finalExamples)):]
maxN=10
maxAcc=0
#for n in [10,30,100]:
for n in [10,30,100]:
#	for b in [1,5,10,25,50,75,100,125,150,175,200]:
	c,g=ma.prepTreeLearner(N=n,k=11,binSize=25)
	acc=ma.acc()
	print("N:",n,"C:",c,"G:",g,"Acc:",acc)
	if acc[0]>maxAcc:
		maxAcc=acc[0]
		maxN=n
ma.prepTreeLearner(N=maxN,k=11,binSize=25)
print("Max N:",maxN)
print(ma.acc())
pl,rl,f1l,pt,rt,f1t=ma.f1Score()
print("For madelon data training set")
print("Precision:",pl)
print("Recall:",rl)
print("F1:",f1l)
print("For madelon data test set")
print("Precision:",pt)
print("Recall:",rt)
print("F1:",f1t)

print("If we increase k from 11 to 100")
ma.prepTreeLearner(N=maxN,k=100,binSize=25)
print("Max N:",maxN)
print(ma.acc())
pl,rl,f1l,pt,rt,f1t=ma.f1Score()
print("For madelon data training set")
print("Precision:",pl)
print("Recall:",rl)
print("F1:",f1l)
print("For madelon data test set")
print("Precision:",pt)
print("Recall:",rt)
print("F1:",f1t)












