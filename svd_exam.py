import svdRec
from numpy import *
import matplotlib.pyplot as plt

set_printoptions(precision=3)

####################### based on Items and SVD  ##############

'''
##### based on Items
myMat=mat(svdRec.loadExData())
print("myMat is\n {}".format(myMat))
print(svdRec.recommend(myMat,2,estMethod=svdRec.standEst))

##### SVD
U, Sigma, VT = linalg.svd(myMat)
rRank=svdRec.calRank(Sigma,0.9)
print("rank of Sigma matrix is: {}".format(rRank))
print(svdRec.recommend(myMat,2,estMethod=svdRec.svdEst))

####################### SVD example ##############
'''
data=mat(svdRec.loadExData2())
print("original matrix:\n {}".format(data))

U, Sigma, VT = linalg.svd(data)
print("U matrix: \n {}".format(U))
print("Sigma matrix: \n {}".format(Sigma))
print("VT matrix: \n {}".format(VT))

rRank=svdRec.calRank(Sigma,0.9)
print("rank of Sigma matrix is: {}".format(rRank))

nSig=Sigma*eye(len(Sigma))
oriM = data
plt.plot(abs(oriM-U*nSig*VT))
plt.ylabel('|Original Matrix - U*Sigma*VT|')

plt.show()

rRank=svdRec.calRank(Sigma,0.9)
nMatrix=U[:,:rRank]*nSig[:rRank,:rRank]*VT[:rRank,:]

plt.plot(abs(oriM-nMatrix))
plt.ylabel('|Original Matrix - U*Sigma*VT (SVD: rank={})|'.format(rRank))

print("U*Sigma*VT \n {} \n \n (SVD: rank={}):".format(nMatrix,rRank))
print("original matrix - U*Sigma*VT(SVD: rank={}) \n {}:".format(rRank,oriM-nMatrix))
plt.show()
