import os
from scipy.ndimage import imread
import numpy as np

class Sample():
    def __init__(self, filepath, classId, sampleId):
        self.filepath = filepath
        self.classId = classId
        self.sampleId = sampleId
        
        
    def load(self):
        # flatten to gray scale
        I = imread(self.filepath,flatten=True)
        return I
        
    def __str__(self):
        return '{}, {}, {}'.format(self.filepath, self.classId, self.sampleId)


class Character():
    def __init__(self, classId):
        self.samples = []
        self.classId = classId
        
    def add(self, sample):
        self.samples.append(sample)
        
    def load(self, n=1):
        return [sample.load() for sample in np.random.choice(self.samples, n, replace=False)]
        
        
class Omniglot():
    def __init__(self):
        # generate list of omniglot images on disk
        self.trainChars = {}
        self.testChars = {}
        
        for root, dirnames, filenames in os.walk('images_background'):
            for filename in filenames:
                classId = int(filename[:4])
                sampleId = int(filename[5:7])
                
                sample = Sample(os.path.join(root, filename), classId, sampleId)
                if classId not in self.trainChars:
                    self.trainChars[classId] = Character(classId)
                    
                self.trainChars[classId].add(sample)
                
                
        for root, dirnames, filenames in os.walk('images_evaluation'):
            for filename in filenames:
                classId = int(filename[:4])
                sampleId = int(filename[5:7])
                
                sample = Sample(os.path.join(root, filename), classId, sampleId)
                if classId not in self.testChars:
                    self.testChars[classId] = Character(classId)
                    
                self.testChars[classId].add(sample)
                
    def _GetBatch(self, sampleDict, batchSize, numClasses=1, samplesPerChar=1):
    
        assert(batchSize <= numClasses * samplesPerChar)
    
        # generate random sample
        samples = np.random.choice(list(sampleDict.keys()), numClasses)
        
        seen = set()
        
        x = []
        y = []
        
        for s in samples:
            character = sampleDict[s]
            x.extend(character.load(samplesPerChar))
            y.extend([character.classId] * samplesPerChar)
            
            
        zipped = list(zip(x, y))
        np.random.shuffle(zipped)
            
        x, y = list(zip(*zipped))
        x = list(x)
        y = list(y)
        
        return x[:batchSize], y[:batchSize]
        
    
    def TrainBatch(self, batchSize, classes=1, samples=1):
        return self._GetBatch(self.trainChars, batchSize, classes, samples)
    
    def TestBatch(self, batchSize, classes=1, samples=1):
        return self._GetBatch(self.testChars, batchSize, classes, samples)
            
# test
if __name__ == '__main__':
    og = Omniglot()
    x, y = og.TrainBatch(100, classes=10, samples=10)
    print(x, y)