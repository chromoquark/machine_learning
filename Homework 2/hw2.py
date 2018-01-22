from random import uniform, shuffle, random, seed
from numpy import mean, std

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
		def __init__(self,initScaler):
			self.w=dict()
			self.initScaler=initScaler
		def __getitem__(self,a):
			if a not in self.w:
				self.w[int(a)]=random()*self.initScaler#uniform(self.initScaler,self.initScaler)
			return self.w[int(a)]
		def __setitem__(self,a,b):
			self.w[int(a)]=b
class perceptron():
	def __init__(self,exampleSet):
		self.exampleSet = exampleSet
		
	
	def run(self,r=1,mu=0,passes=1,w=1,shuff=False,aggressive=False):
		self.w = weights(w)
		self.b=random()*w#uniform(-w,w)
		self.mistakes=0
		self.updates=0
		for i in range(0,passes):
			if shuff:
				shuffle(self.exampleSet)
			for e in self.exampleSet:
				sum=self.b
				for a in e.features:
					sum+=e.features[a]*self.w[a]
				if sum*e.label<0:# or (sum==0 and e.label>0 and useMod):
					self.mistakes+=1
				if sum*e.label<=mu:# or (sum==0 and e.label<0 and useMod):
					self.updates+=1
					if aggressive:
						xSum=0
						for x in e.features:
							xSum+=e.features[x]**2
						r=(mu-e.label*sum)/(xSum+1)
					for a in e.features:
						self.w[a]=self.w[a]+r*e.label*e.features[a]
					self.b+=r*e.label
		return self.w
		
	def test(self,testSet):
		testSet = testSet
		correct = 0
		for t in testSet:
			sum=self.b
			for a in t.features:
				sum+=t.features[a]*self.w[a]
			if sum*t.label>=0:# or (sum==0 and t.label>0):
				correct+=1
		return correct/len(testSet)

	def multipleTests(self,trainSet,testSet,r=1,mu=0,passes=1,w=1,shuff=False,aggressive=False):
		updates=[]
		trainAccuracies=[]
		testAccuracies=[]
		for i in range(0,20):
			self.run(r,mu,passes,w,shuff,aggressive)
			updates.append(self.updates)
			trainAccuracies.append(self.test(trainSet))
			testAccuracies.append(self.test(testSet))
		print("\tUpdates",mean(updates),"+/-",std(updates))
		print("\tTrain set accuracies",mean(trainAccuracies),"+/-",std(trainAccuracies))
		print("\tTest set accuracies",mean(testAccuracies),"+/-",std(testAccuracies))

from numpy import arange
def crossValidateLearningRate(passes,margin=0,shuff=False):
#	ratesToTry=[1,0.7,0.4,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.001,0.0001]
#	ratesToTry=[1,0.3,.1,.03,.01,.003,.001,.0003,.0001]
	ratesToTry=[1,.1,.01,.001,.0001]
	train=exampleSet("./data/a5a.train")
	k=6
#	shuffle(train.set)
	sets=train.kSets(k)
	means = []
	for r in ratesToTry:
		sum = 0
		for i in range(0,k):
			trainer = []
			for j in range(0,k):
				if i!=j:
					for e in sets[j]:
						trainer.append(e)
			shuffle(trainer)
			p = perceptron(trainer)
			p.run(r,passes=passes,mu=margin,shuff=shuff)
			sum += p.test(sets[i])
		means.append(sum/k)
	return ratesToTry[means.index(max(means))]

def crossValidateMargin(r,passes,shuff=False,aggressive=False):
	marginsToTry=[0,0.3,0.7,1,2,3,4,5,6,7,8,9,10]#,6,7,8,9,10,11,12,13,14,15]
	train=exampleSet("./data/a5a.train")
	k=5
	sets=train.kSets(k)
	means = []
	for m in marginsToTry:
		sum = 0
		for i in range(0,k):
			trainer = []
			for j in range(0,k):
				if i!=j:
					for e in sets[j]:
						trainer.append(e)
			p = perceptron(trainer)
			p.run(r,m,passes=passes,shuff=shuff,aggressive=aggressive)
			sum += p.test(sets[i])
		means.append(sum/3)
	return marginsToTry[means.index(max(means))]


def testPerceptron(passes,shuff=False,aggressive=False):
	if not aggressive:
		r = crossValidateLearningRate(passes,shuff)
		print("\tLearning rate for Perceptron",passes,"pass(es): ",r)
		p=perceptron(train.set)
		p.run(passes=passes,shuff=shuff)
		p.multipleTests(train.set,test.set,passes=passes,shuff=shuff)
		m = crossValidateMargin(r,passes,shuff)
		r = crossValidateLearningRate(passes,m,shuff)
		m = crossValidateMargin(r,passes,shuff)
		print("\n\tLearning rate for Perceptron",passes,"pass(es): ",r)
		print("\tMargin for Margin Perceptron: ",m)
		p=perceptron(train.set)
		p.multipleTests(train.set,test.set,passes=passes,shuff=shuff)
	else:
		m=crossValidateMargin(0,passes,shuff,aggressive=aggressive)
		print("\tMargin (mu) for Agressive Perceptron",passes,"passes: ",m)
		p=perceptron(train.set)
		p.multipleTests(train.set,test.set,passes=passes,shuff=shuff,aggressive=aggressive)


def shuffleAndNot(aggressive=False):
	testPerceptron(3,aggressive=aggressive)
	print()
	testPerceptron(5,aggressive=aggressive)
	print("\nShuffling objects during learning")
	testPerceptron(3,shuff=True,aggressive=aggressive)
	print("\nShuffling objects during learning")
	testPerceptron(5,shuff=True,aggressive=aggressive)


table2=exampleSet("./data/table2")
train=exampleSet("./data/a5a.train")
test=exampleSet("./data/a5a.test")


print("3.3.1:")
p = perceptron(table2.set)
print("\tWeight Vector:",p.run(w=0).w,"\n\tBias",p.b,"\n\tMistakes:",p.mistakes,"\n\tTest Accuracy:",p.test(table2.set))


print("3.3.2:")
testPerceptron(1)


print("3.3.3:")
shuffleAndNot()


print("3.3.4")
shuffleAndNot(aggressive=True)
