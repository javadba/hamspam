from multiprocessing import freeze_support

import numpy as np
import math
import datetime,time
from collections import defaultdict
from typing import Dict, List
DictList =  List[Dict[str,int]]

MaxDocs=50  # set lower for testing purposes
MaxFiles=50
RemovePuncts =True
NToPredict=20
# utilities and set up
def p(msg): print(msg)

def formatTimeDelta(td):
  return '%d.%d' %(td.total_seconds() , int(td.microseconds/1000))

def timeit(tag, block):
  import timeit
  def getNow(): return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

  startt = datetime.datetime.now()
  print('Starting %s at %s..' %(tag,getNow()))
  timeit.Timer(block)
  duration = formatTimeDelta(datetime.datetime.now() - startt)
  print('Completed %s at %s with duration=%s secs' %(tag,getNow(), duration))
  
def read(path):
  import codecs
  p('reading %s ..' %path)
  
  with codecs.open(path,'rb', encoding='utf-8',errors='ignore') as f:
    return f.readlines()

def curdir():
  import os
  return os.getcwd()

npa = np.array
import pathlib
baseDir=str(pathlib.Path.home())+ '/Downloads/enron1'
HAM=0
SPM=1
fpaths: List[str]  = []

fpaths.insert(HAM,baseDir + '/ham')
fpaths.insert(SPM,baseDir + '/spam')

def listFiles(dir, maxFiles: int = None):
  import os
  files = [dir+'/'+f for f in os.listdir(dir) if os.path.isfile(dir+'/'+f)]
  if maxFiles:
    return files[:min(len(files),maxFiles)]
  else:
    return files

###
# Text processing utilities
###

# Bag of words
# def bofw(text, caseInsensitive=True) -> dict:
#   import re
#   text = ''.join(text)
#   text = text.lower() if caseInsensitive else text
#   toks: list = filter(lambda x: len(x) > 0, re.sub(r'[^\w\s]','',text).split(' '))
#   # toks: list = text.translate({ord(i):None for i in '\r/!@#$'}).split(' ')
#   from collections import Counter
#   bow: dict = Counter(toks)
#   return bow

shortStops = ['a','and','the','an','of']

# Ngrams that will feed into bag of words
# removeStops = apply stopwords
# minLen = min length of words e.g. if set to 2 will remove single letter tokens
def ngrams(text, caseInsensitive = True, removeStops = False, minLen = 1) -> dict:
  import re
  text = ''.join(text)
  text = text.lower() if caseInsensitive else text
  text = re.sub(r'[^\w\s]','',text) if RemovePuncts else text
  toks: list = list(filter(lambda x: len(x) > 0,re.split('[\\s]+',text)))
  toks = list(map(lambda x: x.strip(),toks))
  toks = [t for t in toks if len(t) >= minLen]
  if removeStops:
    stops = read(baseDir+'/stopwords.txt').split('\n')
    toks = [t for t in toks if t not in stops]
    
  ngs = defaultdict(int)
  for ln in range(1,4):
    for ix in range(0,len(toks)-(ln-1)):
      k = '-'.join(toks[ix:ix+ln])
      ngs[k] = ngs[k] + 1
  return ngs

# Converts a dict (bag of words) into unnormalized wordvec
def wordvec(allWords, bofw):
  allw = allWords
  return npa(list(bofw[w] if w in bofw else 0 for w in allw))

# Basic cosine calc between two unnormalized vecs
def cosineSim(doc1,doc2):
  l1 = np.sum(np.dot(doc1,doc1))
  l2 = np.sum(np.dot(doc2,doc2))
  cos = np.dot(doc1,doc2) / np.sqrt(l1*l2)
  return cos

###
# Algorithms
###


# numpy does not have a groupby .. amazing ..
from typing import List, Dict, Callable
ElemType = float
def groupby(arr: np.array, transformfn: Callable[[ElemType], ElemType],
            selectfn: Callable[[ElemType], ElemType] = None) -> Dict[int, npa]:
  xarr = list(map(transformfn,arr))
  keys, indx = np.unique(xarr, return_inverse=True)
  K = len(keys)
  recsByKey = list()
  for i in range(K):
    if selectfn:
      recsByKey.append([selectfn(x) for x in np.array(arr)[np.where(indx == i)]])
    else:
      recsByKey.append(arr[np.where(indx == i)])
  return {k:v for k,v in list(zip(keys,recsByKey))}

# euclidean distance between 2 vecs
def euclid(p1,p2):
  return np.sqrt((np.sum([(pp1-pp2)**2 for pp1,pp2 in list(zip(p1,p2))])))

# Kmeans given an initial set of centroids (or start with 0's)
from typing import Tuple
MinKmeansLoops = 7
def kmeans(pts: npa, initCentroids: npa = None, K: int = None, maxIters: int = 15) -> List[Tuple[int, float]]:
  W = len(pts[0])  # num Terms
  if not K:
    K = np.sqrt(len(pts)/2) # Recommended from my clustering class and
                             # experience on job showed reasonable
  centroids = list(enumerate(initCentroids
                             if initCentroids
                             else np.array(pts)[np.random.choice(range(len(pts)),K,replace=False)]))
  curcentrs = centroids
  curPtToCentrs = np.zeros(len(pts))
  for loopx in range(maxIters):
    newPtToCentrs = []
    for i in range(len(pts)):
      # newPtToCentrs[i] = curcentrs[np.argmin([euclid(pts[i],centr[1]) for centr in curcentrs])][0]
      if curcentrs is None:
        pass
      centrdists = [euclid(pts[i],centr[1]) for centr in curcentrs]
      newPtToCentrs.append(np.argmin(centrdists))
    
    # reasonable stopping point if noone switched cluster allegiance
    if loopx>= MinKmeansLoops and np.allclose(newPtToCentrs,curPtToCentrs):
      p('after %d loops we hit convergence for kmeans' %loopx)
      break
    curPtToCentrs = newPtToCentrs.copy()
    newcentrsGrps= groupby(list(zip(newPtToCentrs,pts)), lambda y: y[0], lambda y: y[1])
    curcentrs = npa([(grpId,np.mean(centroidPts,axis=0)) for grpId, centroidPts in newcentrsGrps.items()])
    curcentrs = curcentrs[curcentrs[:,0].argsort()]
  return curcentrs

# Find all unique terms in a corpus
def terms(docTerms: DictList) -> list:
  allWords= set()
  for b in docTerms:
    allWords.update(b.keys())
  allWords = sorted(list(allWords))
  return allWords

# Predict the label of a document[-vector] given the Ham/Spam centroids and a KNN count for voting
CentroidType = List[Tuple[str,npa]]
Point = np.array
def predict(pt: Point, centroids: CentroidType, closestK: int = 5) -> str:
  import operator
  from collections import Counter
  distances = [ (id, euclid(pt, centr)) for id,centr in centroids]
  distances.sort(key=operator.itemgetter(1))
  voters = distances[0:closestK]
  votes = Counter([v[0].split(":")[0] for v in voters]).most_common()
  return votes[0][0]

def accuracy(acts,expecteds):
  return float(np.count_nonzero(npa(expecteds) == npa(acts))) / len(expecteds)
  
#########
if __name__ == '__main__':

  import dask
  from dask import bag as daskbag

  from distributed import Client, LocalCluster
  # cluster = LocalCluster(n_workers=8, threads_per_worker=2)
  cluster = LocalCluster(n_workers=1, threads_per_worker=1)
  client = Client(cluster)
  
  freeze_support()
  p('STARTING job at %s ..' %time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
  startt = datetime.datetime.now()
  
  # Read the data files
  bows: Dict = {}  # bags of words in form of lists of Dicts with a list for Ham's another for Spam's
  for i in [HAM,SPM]:
    fnames = daskbag.from_sequence(listFiles(fpaths[i],MaxFiles))
    files = fnames.map(read)
    bows[i] = files.map(ngrams)
    bows[i].compute()

  # Create the word dimension
  allTerms = terms(bows.values())
  
  # Apply word dimension to all docs in both ham/spam
  from functools import partial
  wvecs = []
  for hamspm in [HAM,SPM]:
    bow = daskbag.from_sequence(bows[hamspm][:MaxDocs])
    wordvec1 = partial(wordvec, allTerms)
    wvec = bow.map(wordvec1(allTerms))
    p('Computing wordvec for hamspm=%d ..' %hamspm)
    wvec.compute()
    wvecs.append(wvec)
    
  for hamspm in [HAM,SPM]:
    centroids = wvecs[hamspm].map(kmeans)
    centroids.compute()
    
  labeledCentroids = np.vstack(np.array([[('%d:%d' %(hamspm,idptt[0]),idptt[1]) for idptt in idpt]
                                         for hamspm,idpt in list(zip([HAM,SPM],centroids))]))
  
  # try out first 10 of each ham, spam - see if they get properly predicted
  preds = [[predict(p, labeledCentroids, 5) for p in wvecs[hamspm][:NToPredict]] for hamspm in [HAM,SPM]]
  
  p(repr(preds))
  
  acc = accuracy(preds, np.concatenate((np.zeros(NToPredict),np.ones(NToPredict))))
  p('COMPLETED job with Accuracy=%f at %s duration: %s secs' %(acc,
                                                               time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                              formatTimeDelta(datetime.datetime.now() - startt)))
  
  # now let's re-test against the full input dataset to see how ham/spam fall out
    
  
