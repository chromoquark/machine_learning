from __future__ import division
from random import uniform, shuffle, random, seed
from numpy import mean, std, arange
from math import exp, log
from matplotlib import pyplot


class example():
	def __init__(self,a):
		a=a.split(" ")
		self.label=int(a[0])
		del a[0]
		self.features=dict()
		for i in a:
			if i=="\n": continue
			b=i.split(":")
			self.features[int(b[0])]=int(b[1])
			self.features[-1]=1
	def __getitem__(self,a):
		if a in self.features: return self.features[int(a)]
		return 0
	def __setitem__(self,a,b):
		self.features[int(a)]=int(b)

class exampleSet():
	def __init__(self,file):
		examples=[]
		for line in open(file,'r'):
			examples.append(example(line))
		self.set = examples
	def kSets(self,k):
		n=len(self.set)//k
		sets = []
		for i in range(0,k-1):
			sets.append(self.set[i*n:(i+1)*n])
		sets.append(self.set[(k-1)*n:len(self.set)+1])
		return sets


class weights():
		def __init__(self):
			self.w=dict()
		def __getitem__(self,a):
			if a not in self.w:
				self.w[int(a)]=0
			return self.w[int(a)]
		def __setitem__(self,a,b):
			self.w[int(a)]=b
class perceptron():
	def __init__(self,exampleSet):
		self.exampleSet = exampleSet
		
	def run(self,gamma=1,epoch=1,sigma=1,C=1,testSet=None):
		seed(17071955)
		self.w = weights()
		objectiveValues=[]
		accuracy=[]
		t=0
		for i in range(epoch):
			shuffle(self.exampleSet)
			for e in self.exampleSet:
				gammat=gamma/(1+gamma*t/C)
				t+=1
				sum=0
				for a in e.features:
					sum+=e.features[a]*self.w[a]

				loss=gammat*e.label/(1+exp(e.label*sum))

				regularizer=1-2*gammat/sigma**2
				for a in e.features:
					self.w[a]=regularizer*self.w[a]+loss*e.features[a]
			if testSet!=None:
				objectiveValues.append(self.objectiveVal(sigma))
				accuracy.append(self.test(testSet))
		return self.w,objectiveValues,accuracy
		
	def test(self,testSet):
		correct = 0
		numNeg=0
		positives=0
		for t in testSet:
			sum=0
			for a in t.features:
				sum+=t.features[a]*self.w[a]
			predictionNum=1/(1+exp(-sum))
			if predictionNum>=0.5: prediction=1
			else: prediction=-1
			
			if prediction<=0:
				positives+=1

			if prediction*t.label>=0:			
				correct+=1

		return correct/len(testSet)
	
	def objectiveVal(self,sigma):
		o1=0
		for a in self.w.w:
			o1+=self.w[a]
		o2=0
		for e in self.exampleSet:
			sum=0
			for a in e.features:
				sum+=e.features[a]*self.w[a]
			o2+=log(1+exp(-e.label*sum))
		return o1/sigma**2+o2

def crossValidate(train,varToTest,gamma=.1,sigma=1,epoch=10,C=1):
	param=dict(gamma=gamma,epoch=epoch,sigma=sigma,C=C)
	if varToTest=="gamma":
		toTry=[0.9]
		for i in range(1,4):
			toTry.append(10**-i)
	if varToTest=="epoch":
		toTry=[]
		for i in range(0,6):
			toTry.append(2**i)
	if varToTest=="sigma":
		toTry=[]
		for i in range(0,8):
			toTry.append(1.5*5**i)
	if varToTest=="C":
		toTry=[]
		for i in range(-5,6):
			toTry.append(10**-i)


	train=exampleSet("./data/a5a.train")
	k=6
	sets=train.kSets(k)
	means = []
	for r in toTry:
		param[varToTest]=r
		total = 0
		for i in range(0,k):
			trainer = []
			for j in range(0,k):
				if i!=j:
					for e in sets[j]:
						trainer.append(e)
			shuffle(trainer)
			p = perceptron(trainer)
			p.run(**param)
			test = p.test(sets[i])
			total += test
		means.append(total/k)
	return toTry[means.index(max(means))]




table2=exampleSet("./data/table2")
train=exampleSet("./data/a5a.train")
test=exampleSet("./data/a5a.test")

gamma=0.9
epoch=16
sigma=5
C=1000

for i in range(0,1):
	sigma=crossValidate(train,"sigma",gamma,sigma,epoch,C)
	gamma=crossValidate(train,"gamma",gamma,sigma,epoch,C)
	epoch=crossValidate(train,"epoch",gamma,sigma,epoch,C)
	C=crossValidate(train,"C",gamma,sigma,epoch,C)
	print("Gamma:",gamma,"\tEpoch:",epoch,"\tSigma:",sigma,"\tC:",C)

p = perceptron(train.set)
w,obj,acc = p.run(gamma=gamma,epoch=epoch,sigma=sigma,C=C,testSet=test.set)

f, axarr = pyplot.subplots(2,sharex="row")
axarr[0].plot(arange(0,epoch,1)+1,obj,"r")
pyplot.ylabel("Objective")
axarr[1].plot(arange(0,epoch,1)+1,acc,"g")
pyplot.ylabel("Accuracy")
pyplot.xlabel("Epoch")
print(obj)
print(acc)
pyplot.show()
