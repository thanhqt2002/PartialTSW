#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 14:25:29 2022
@author: Yikun Bai yikun.bai@Vanderbilt.edu 
"""
import torch 
import os
import sys
import numpy as np
import time

from .cvpr23_bai import * 
import matplotlib.pyplot as plt
from partial import partial_ot_1d, partial_ot_1d_elbow
from sliced import PartialSWGG
from tqdm import tqdm

@nb.njit(['float64[:,:](int64,int64,int64)'],fastmath=True,cache=True)
def random_projections(d,n_projections,Type=0):
    '''
    input: 
    d: int 
    n_projections: int

    output: 
    projections: d*n torch tensor

    '''
#    np.random.seed(0)
    if Type==0:
        Gaussian_vector=np.random.normal(0,1,size=(d,n_projections)) #.astype(np.float64)
        projections=Gaussian_vector/np.sqrt(np.sum(np.square(Gaussian_vector),0))
        projections=projections.T

    elif Type==1:
        r=np.int64(n_projections/d)+1
        projections=np.zeros((d*r,d)) #,dtype=np.float64)
        for i in range(r):
            H=np.random.randn(d,d) #.astype(np.float64)
            Q,R=np.linalg.qr(H)
            projections[i*d:(i+1)*d]=Q
        projections=projections[0:n_projections]
    return projections





@nb.njit(['Tuple((float64[:],int64[:,:]))(float64[:,:],float64[:,:],float64[:])'],parallel=True,fastmath=True,cache=True)
def opt_plans(X_sliced,Y_sliced,Lambda_list):
    N,n=X_sliced.shape
#    Dtype=type(X_sliced[0,0])
    plans=np.zeros((N,n),np.int64)
    costs=np.zeros(N,np.float64)
    for i in nb.prange(N):
        X_theta=X_sliced[i]
        Y_theta=Y_sliced[i]
        Lambda=Lambda_list[i]
        # M=cost_matrix(X_theta,Y_theta)
        obj,phi,psi,piRow,piCol=solve_opt(X_theta,Y_theta,Lambda)
        cost=obj
        L=piRow
        plans[i]=L
        costs[i]=cost
    return costs,plans





@nb.njit(['(float64[:,:],float64[:,:],float64[:,:],float64[:])'],cache=True)
def X_correspondence(X,Y,projections,Lambda_list):
    N,d=projections.shape
    n=X.shape[0]
    Lx_org=arange(0,n)
    for i in range(N):
        theta=projections[i]
        X_theta=np.dot(theta,X.T)
        Y_theta=np.dot(theta,Y.T)
        X_indice=X_theta.argsort()
        Y_indice=Y_theta.argsort()
        X_s=X_theta[X_indice]
        Y_s=Y_theta[Y_indice]
        Lambda=Lambda_list[i]
        # M=cost_matrix(X_s,Y_s)
        obj,phi,psi,piRow,piCol=solve_opt(X_s,Y_s,Lambda)
#        Cost,L=o(X_s,Y_s,Lambda)
        
        L=piRow
        L=recover_indice(X_indice,Y_indice,L)
        #move X
        Lx=Lx_org.copy()
        Lx=Lx[L>=0]
        if Lx.shape[0]>=1:
            Ly=L[L>=0]
#            dim=Ly.shape[0]
            X_take=X_theta[Lx]
            Y_take=Y_theta[Ly]
            X[Lx]+=np.expand_dims(Y_take-X_take,1)*theta
            


 

@nb.njit(['(float64[:,:],float64[:,:],float64[:,:])'],cache=True)
def X_correspondence_pot(X,Y,projections):
    N,d=projections.shape
    n=X.shape[0]
    for i in range(N):
        theta=projections[i]
        X_theta=np.dot(theta,X.T)
        Y_theta=np.dot(theta,Y.T)
        X_indice=X_theta.argsort()
        Y_indice=Y_theta.argsort()
        X_s=X_theta[X_indice]
        Y_s=Y_theta[Y_indice]
        # M=cost_matrix(X_s,Y_s)
        cost,L=pot(X_s,Y_s)
        L=recover_indice(X_indice,Y_indice,L)
        X_take=X_theta
        Y_take=Y_theta[L]
        X+=np.expand_dims(Y_take-X_take,1)*theta
    return X


    

@nb.njit(['Tuple((float64,int64[:,:],float64[:,:],float64[:,:]))(float64[:,:],float64[:,:],float64[:])'],parallel=True,fastmath=True,cache=True)
def opt_plans_64(X,Y,Lambda_list):
    n,d=X.shape
    n_projections=Lambda_list.shape[0]
    projections=random_projections(d,n_projections,0)
    X_projections=projections.dot(X.T)
    Y_projections=projections.dot(Y.T)
    opt_plan_X_list=np.zeros((n_projections,n),dtype=np.int64)
    #opt_plan_Y_list=np.zeros((n_projections,n),dtype=np.int64)
    opt_cost_list=np.zeros(n_projections)
    for (epoch,(X_theta,Y_theta,Lambda)) in enumerate(zip(X_projections,Y_projections,Lambda_list)):
        X_indice=X_theta.argsort()
        Y_indice=Y_theta.argsort()
        X_s=X_theta[X_indice]
        Y_s=Y_theta[Y_indice]
        # M=cost_matrix(X_s,Y_s)
        obj,phi,psi,piRow,piCol=solve_opt(X_s,Y_s,Lambda)
        
        L1=recover_indice(X_indice,Y_indice,piRow)
        #L2=recover_indice(Y_indice,X_indice,piCol)
        opt_cost_list[epoch]=obj
        opt_plan_X_list[epoch]=L1
        #opt_plan_Y_list[epoch]=L2
        #sopt_dist=np.sum(opt_cost_list)/n_projections
        sopt_dist=opt_cost_list.sum()/n_projections
    return sopt_dist,opt_plan_X_list,X_projections,Y_projections

def opt_cost_from_plans(X_projections,Y_projections,Lambda_list,opt_plan_X_list,cache=True):
    n_projections,n=X_projections.shape
    n_projections,m=Y_projections.shape
    opt_cost_list=np.zeros(n_projections)
    for (epoch,(X_theta,Y_theta,Lambda,opt_plan)) in enumerate(zip(X_projections,Y_projections,Lambda_list,opt_plan_list)):
        Domain=opt_plan>=0
        Range=opt_plan[Domain]
        X_select=X_theta[Domain]
        Y_select=Y_theta[Range]
        trans_cost=np.sum(cost_function(X_select,Y_select))
        mass_panalty=Lambda*(m+n-2*Domain.sum())
        opt_cost_list[epoch]=trans_cost+mass_panalty
    return opt_cost_list

@nb.njit(cache=True,fastmath=False,parallel=True)
def cost_matrix(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
#    XT=np.expand_dims(X,1)
    n,m=X.shape[0],Y.shape[0]
    M=np.zeros((n,m))
    for i in nb.prange(n):
        for j in nb.prange(m):
            M[i,j]=(X[i]-Y[j])**p   
    return M


@nb.njit(fastmath=True,cache=True)
def cost_matrix_d(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
    n,d=X.shape
    m,d=Y.shape
    M=np.empty((n,m))
    for i in nb.prange(n):
        M[i]=np.sum((X[i]-Y)**p,1)
    return M






@nb.njit(['int64[:](int64,int64)'],fastmath=True,cache=True)
def arange(start,end):
    n=end-start
    L=np.zeros(n,np.int64)
    for i in range(n):
        L[i]=i+start
    return L



@nb.njit(['Tuple((int64,int64))(int64[:])'],cache=True)
def unassign_y(L1):
    '''
    Parameters
    ----------
    L1 : n*1 list , whose entry is 0,1,2,...... 
            transporportation plan. L[i]=j denote we assign x_i to y_j, L[i]=-1, denote we destroy x_i. 
            if we ignore -1, L1 must be in increasing order 
            make sure L1 do not have -1 and is not empty, otherwise there is mistake in the main loop.  


    Returns
    -------
    i_act: integer>=0 
    j_act: integer>=0 or -1    
    j_act=max{j: j not in L1, j<L1[end]} If L1[end]=-1, there is a bug in the main loop. 
    i_act=min{i: L[i]>j_act}.
    
    Eg. input: L1=[1,3,5]
    return: 2,4
    input: L1=[2,3,4]
    return: 0,1
    input: L1=[0,1,2,3]
    return: 0,-1
    
    '''
    n=L1.shape[0]
    j_last=L1[n-1]
    i_last=L1.shape[0]-1 # this is the value of k-i_start
    for l in range(n):
        j=j_last-l
        i=i_last-l+1
        if j > L1[n-1-l]:
            return i,j
    j=j_last-n
    if j>=0:
        return 0,j
    else:       
        return 0,-1



@nb.njit(['Tuple((int64,int64))(int64[:])'],cache=True)
def unassign_y_nb(L1):
    '''
    Parameters
    ----------
    L1 : n*1 list , whose entry is 0,1,2,...... 
            transporportation plan. L[i]=j denote we assign x_i to y_j, L[i]=-1, denote we destroy x_i. 
            if we ignore -1, L1 must be in increasing order 
            make sure L1 do not have -1 and is not empty, otherwise there is mistake in the main loop.  


    Returns
    -------
    i_act: integer>=0 
    j_act: integer>=0 or -1    
    j_act=max{j: j not in L1, j<L1[end]} If L1[end]=-1, there is a bug in the main loop. 
    i_act=min{i: L[i]>j_act}.
    
    Eg. input: L1=[1,3,5]
    return: 2,4
    input: L1=[2,3,4]
    return: 0,1
    input: L1=[0,1,2,3]
    return: 0,-1
    
    '''
    
    j_last=L1[-1]
    n=L1.shape[0]
    L_range=arange(j_last-n+1,j_last+1)
    L_dif=np.where(L_range-L1>0)[0]
    if L_dif.shape[0]==0:
        return 0, L1[0]-1
    else:
        i_act=L_dif[-1]+1
        j_act=L_range[i_act-1]
    return i_act,j_act



@torch.jit.script   
def recover_indice_T(indice_X,indice_Y,L):
    '''
    input:
        indice_X: n*1 float torch tensor, whose entry is integer 0,1,2,....
        indice_Y: m*1 float torch tensor, whose entry is integer 0,1,2,.... 
        L: n*1 list, whose entry could be 0,1,2,... and -1.
        L is the original transportation plan for sorted X,Y 
        L[i]=j denote x_i->y_j and L[i]=-1 denote we destroy x_i. 
        If we ignore -1, it must be in increasing order  
    output:
        mapping_final: the transportation plan for original unsorted X,Y
        
        Eg. X=[2,1,3], indice_X=[1,0,2]
            Y=[3,1,2], indice_Y=[1,2,0]
            L=[0,1,2] which means the mapping 1->1, 2->2, 3->3
        return: 
            L=[2,1,0], which also means the mapping 2->2, 1->1,3->3.
    
    '''
    device=indice_X.device.type
    n=L.shape[0]
#    indice_Y_mapped=torch.tensor([indice_Y[i] if i>=0 else -1 for i in L],device=device)
    indice_Y_mapped=torch.where(L>=0,indice_Y[L],-1).to(device) 
    mapping=torch.stack([indice_X,indice_Y_mapped])
    mapping_final=mapping[1].take(mapping[0].argsort())
    return mapping_final



@nb.njit(['int64[:](int64[:],int64[:],int64[:])'],cache=True)
def recover_indice(indice_X,indice_Y,L):
    '''
    input:
        indice_X: n*1 float torch tensor, whose entry is integer 0,1,2,....
        indice_Y: m*1 float torch tensor, whose entry is integer 0,1,2,.... 
        L: n*1 list, whose entry could be 0,1,2,... and -1.
        L is the original transportation plan for sorted X,Y 
        L[i]=j denote x_i->y_j and L[i]=-1 denote we destroy x_i. 
        If we ignore -1, it must be in increasing order  
    output:
        mapping_final: the transportation plan for original unsorted X,Y
        
        Eg. X=[2,1,3], indice_X=[1,0,2]
            Y=[3,1,2], indice_Y=[1,2,0]
            L=[0,1,2] which means the mapping 1->1, 2->2, 3->3
        return: 
            L=[2,1,0], which also means the mapping 2->2, 1->1,3->3.
    
    '''
    n=L.shape[0]
    indice_Y_mapped=np.where(L>=0,indice_Y[L],-1)
    mapping=np.stack((indice_X,indice_Y_mapped))
    mapping_final=mapping[1].take(mapping[0].argsort())
    return mapping_final


@nb.njit(fastmath=True,cache=True)
def closest_y_M(M):
    '''
    Parameters
    ----------
    x : float number, xk
    Y : m*1 float np array, 

    Returns
    -------
    min_index : integer >=0
        argmin_j min(x,Y[j])  # you can also return 
    min_cost : float number 
        Y[min_index]

    '''
    n,m=M.shape
    argmin_Y=np.zeros(n,np.int64)
    for i in range(n):
        argmin_Y[i]=M[i,:].argmin()
    return argmin_Y


@nb.njit(['int64[:,:](int64[:],int64)'],fastmath=True,cache=True)
def array_to_matrix(L,m):
    '''
    Parameters
    ----------
    L : n*1 tensor, whose entries is 0,1,2,.... or -1
    
    m : integer >=0 
    
    Returns
    -------
    plan : n*m matrix
    plan[i,j]=1 if L[i]=j and j>=0
    otherwise, plan[i,j]=0
 

    '''
    n=L.shape[0]
    plan=np.zeros((n,m),np.int64)
    
    Ly=L[L>=0]
    Lx=arange(0,n)
    Lx=Lx[L>=0]
    for i in Lx:
        plan[i,L[i]]=1
    return plan

@nb.njit(['int64[:](int64[:,:])'],fastmath=True,cache=True)
#@nb.njit(fastmath=True)
def L_to_pi(L_lp):
    '''
    Parameters
    ----------
    L : n*1 tensor, whose entries is 0,1,2,.... or -1
    
    m : integer >=0 
    
    Returns
    -------
    plan : n*m matrix
    plan[i,j]=1 if L[i]=j and j>=0
    otherwise, plan[i,j]=0
 

    '''
    n,m=L_lp.shape
    L=np.full(n,-1,np.int64)
    for i in range(n):
        indexes=np.where(L_lp[i,:]>=0.5)[0]
        if indexes.shape[0]==1:
            L[i]=indexes[0]
        elif indexes.shape[0]>=2:
            print('error')
    return L


@nb.njit(['(float64[:])(float64[:],float64[:],int64)'],fastmath=True,cache=True)
def Gaussian_mixture(mu_list, variance_list,n):
    N=mu_list.shape[0]
    indices=np.random.randint(0,N,n)
    X=np.zeros(n)
    for i in range(n):
        X[i]=np.random.normal(mu_list[indices[i]],variance_list[indices[i]])
    return X

@nb.njit(['(float32[:])(float32[:],float32[:],int64)'],fastmath=True,cache=True)
def Gaussian_mixture_32(mu_list, variance_list,n):
    N=mu_list.shape[0]
    indices=np.random.randint(0,N,n)
    X=np.zeros(n,dtype=np.float32)
    for i in range(n):
        X[i]=np.float32(np.random.normal(mu_list[indices[i]],variance_list[indices[i]]))
    return X

# def get_swiss(N=100,a = 4,r_min = 0.1,r_max = 1): 
#     """
#     generate swiss shape data
#     parameters: 
#     -------
#     N : int64 
#     a: float or int64
#     r_min: float
#     r_max: float

#     returns:
#     ------
#     X: numpy array, shape (N,2), float64 
#     """

#   theta = np.linspace(0, a * np.pi, N)
#   r = np.linspace(r_min, r_max, N)
#   X = np.stack([r * np.cos(theta),r * np.sin(theta)],1)
#   return X


def rotation_matrix(theta):
    """
    generate (2,2) rotation matrix
    
    Parameter:
    ------
    theta : float
    
    Returns: 
    -------
    torch.tensor shape (2,2) float 
    """
    return torch.stack([torch.cos(theta).reshape([1]),torch.sin(theta).reshape([1]),
            -torch.sin(theta).reshape([1]),torch.cos(theta).reshape([1])]).reshape([2,2])





def rotation_matrix_3d_x(theta_x):
    """
    generate (3,3) rotation matrix along x-axis 
    
    Parameter:
    -----
    theta: float
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float 
    """
    device=theta_x.device.type
    rotation_x=torch.zeros((3,3),dtype=torch.float64,device=device)
    rotation_x[1,1]=torch.cos(theta_x)
    rotation_x[1,2]=-torch.sin(theta_x)
    rotation_x[2,1]=torch.sin(theta_x)
    rotation_x[2,2]=torch.cos(theta_x)
    rotation_x[0,0]=1.0
    return rotation_x


def rotation_matrix_3d_y(theta_y):
    """
    generate (3,3) rotation matrix along y-axis 
    
    Parameter:
    -----
    theta: float
    
    Returns: 
    ------
    torch.tensor shape (3,3) float 
    """
        
    device=theta_y.device.type
    rotation_y=torch.zeros((3,3),dtype=torch.float64,device=device)
    rotation_y[0,0]=torch.cos(theta_y)
    rotation_y[0,2]=torch.sin(theta_y)
    rotation_y[2,0]=-torch.sin(theta_y)
    rotation_y[2,2]=torch.cos(theta_y)
    rotation_y[1,1]=1.0
    return rotation_y

def rotation_matrix_3d_z(theta_z):
    """
    generate (3,3) rotation matrix along z-axis 
    
    Parameter:
    -----
    theta: float
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float 
    """
        
    device=theta_z.device.type
    rotation_z=torch.zeros((3,3),dtype=torch.float64,device=device)
    rotation_z[0,0]=torch.cos(theta_z)
    rotation_z[0,1]=-torch.sin(theta_z)
    rotation_z[1,0]=torch.sin(theta_z)
    rotation_z[1,1]=torch.cos(theta_z)
    rotation_z[2,2]=1.0
    return rotation_z

def rotation_matrix_3d(theta,order='re'):
    
    """
    generate (3,3) rotation matrix 
    
    Parameter:
    -----
    theta: torch tensor (3,) float
    order: string "re" or "in" 
          "in" roation with respect to x-axis, then y-axis, then z-axis
          "re" rotation with rspect to z-axis, then y-axis, then x-axis 
    Returns: 
    ------
    M: torch.tensor shape (3,3) float 
    """
        
    theta_x,theta_y,theta_z=theta
    rotatioin_x=rotation_matrix_3d_x(theta_x)
    rotatioin_y=rotation_matrix_3d_y(theta_y)
    rotatioin_z=rotation_matrix_3d_z(theta_z)
    if order=='in':
        rotation_3d=torch.linalg.multi_dot((rotatioin_z,rotatioin_y,rotatioin_x))
    elif order=='re':
        rotation_3d=torch.linalg.multi_dot((rotatioin_x,rotatioin_y,rotatioin_z))
    return rotation_3d

def rotation_3d_2(theta,order='re'):
    """
    generate (3,3) rotation matrix 
    
    Parameter:
    -----
    theta: torch tensor (3,) float
    order: string "re" or "in" 
          "in" roation with respect to x-axis, then y-axis, then z-axis
          "re" rotation with rspect to z-axis, then y-axis, then x-axis 
    Returns: 
    ------
    M: torch.tensor shape (3,3) float 
    """
        
    cos_x,cos_y,cos_z=torch.cos(theta)
    sin_x,sin_y,sin_z=torch.sin(theta)

    if order=='re':
        M=rotation_re(cos_x,sin_x,cos_y,sin_y,cos_z,sin_z)
    elif order=='in':
        M=rotation_in(cos_x,sin_x,cos_y,sin_y,cos_z,sin_z)
    return M

def rotation_re(cos_x,sin_x,cos_y,sin_y,cos_z,sin_z):
    """
    generate (3,3) rotation matrix along  
    
    Parameter:
    -----
    cos_x,sin_x: float,float
                cos(x), sin(x) for some angle x
    cos_y,sin_y: float, float
                cos(y), sin(y) for some angle y
    cos_z,sin_z: float, float
                cos(z), sin(z) for some angle z
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float
             rotation with rspect to z-axis, then y-axis, then x-axis
    """
    
    M=torch.zeros((3,3),dtype=torch.float64)
    M[0,0]=cos_y*cos_z
    M[0,1]=-cos_y*sin_z
    M[0,2]=sin_y
    M[1,0]=sin_x*sin_y*cos_z+cos_x*sin_z
    M[1,1]=-sin_x*sin_y*sin_z+cos_x*cos_z
    M[1,2]=-sin_x*cos_y
    M[2,0]=-cos_x*sin_y*cos_z+sin_x*sin_z
    M[2,1]=cos_x*sin_y*sin_z+sin_x*cos_z 
    M[2,2]=cos_x*cos_y
    return M

def rotation_in(cos_x,sin_x,cos_y,sin_y,cos_z,sin_z):
    """
    generate (3,3) rotation matrix along  
    
    Parameter:
    -----
    cos_x,sin_x: float,float
                cos(x), sin(x) for some angle x
    cos_y,sin_y: float, float
                cos(y), sin(y) for some angle y
    cos_z,sin_z: float, float
                cos(z), sin(z) for some angle z
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float
             rotation with rspect to x-axis, then y-axis, then z-axis
    """
    M=torch.zeros((3,3))
    M[0,0]=cos_y*cos_z
    M[0,1]=-cos_x*sin_z+sin_x*sin_y*cos_z
    M[0,2]=sin_x*sin_z+cos_x*sin_y*cos_z
    M[1,0]=cos_y*sin_z
    M[1,1]=cos_x*cos_z+sin_x*sin_y*sin_z
    M[1,2]=-sin_x*cos_z+cos_x*sin_y*sin_z
    M[2,0]=-sin_y
    M[2,1]=sin_x*cos_y
    M[2,2]=cos_x*cos_y
    return M


    

    

@nb.njit(['float64[:](float64[:,:])'],fastmath=True,cache=True)
def vec_mean(X):
    """
    return X.mean(1) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot64
    
    Return:
    --------
    mean: numpy array, shape (d,), float64 
    
    
    """
    n,d=X.shape
    mean=np.zeros(d,dtype=np.float64)
    for i in nb.prange(d):
        mean[i]=X[:,i].mean()
    return mean
        


    
    
@nb.njit(['Tuple((float64[:,:],float64))(float64[:,:],float64[:,:])'],cache=True)
def recover_rotation(X,Y):
    """
    return the optimal rotation, scaling based on the correspondence (X,Y) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot64, target
    Y: numpy array, shape (n,d), flaot64, source
    
    Return:
    --------
    rotation: numpy array, shape (d,d), float64 
    scaling: float64 
    
    """

        
    n,d=X.shape
    X_c=X-vec_mean(X)
    Y_c=Y-vec_mean(Y)
    YX=Y_c.T.dot(X_c)
    U,S,VT=np.linalg.svd(YX)
    R=U.dot(VT)
    diag=np.eye(d,dtype=np.float64)
    diag[d-1,d-1]=np.linalg.det(R.T)
    rotation=U.dot(diag).dot(VT)
    scaling=np.sum(np.abs(S.T))/np.trace(Y_c.T.dot(Y_c))
    return rotation,scaling



@nb.njit(['Tuple((float64[:,:],float64[:]))(float64[:,:],float64[:,:])'],fastmath=True,cache=True)
def recover_rotation_du(X,Y):
    """
    return the optimal rotation, scaling based on the correspondence (X,Y) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot64, target
    Y: numpy array, shape (n,d), flaot64, source
    
    Return:
    --------
    rotation: numpy array, shape (d,d), float64 
    scaling: numpy array, shape (d,) float64 
    
    """
    
    n,d=X.shape
    X_c=X-vec_mean(X)
    Y_c=Y-vec_mean(Y)
    YX=Y_c.T.dot(X_c)
    U,S,VT=np.linalg.svd(YX)
    R=U.dot(VT)
    diag=np.eye(d,dtype=np.float64)
    diag[d-1,d-1]=np.linalg.det(R)
    rotation=U.dot(diag).dot(VT)
    E_list=np.eye(d,dtype=np.float64)
    scaling=np.zeros(d,dtype=np.float64)
    for i in range(d):
        Ei=np.diag(E_list[i])
        num=0
        denum=0
        for j in range(d):
            num+=X_c[j].dot(rotation.T).dot(Ei).dot(Y_c[j])
            denum+=Y_c[j].dot(Ei).dot(Y_c[j])
        scaling[i]=num/denum
    return rotation,scaling





# our method
#@nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:]))(float64[:,:],float64[:,:],int64,int64)'])
# @nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:],float64[:]))(float64[:,:],float64[:,:],int64,int64)'],cache=True)
def sopt_main(S,T,n_iterations,N0):
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float32
        source data 
    T: (n,d) numpy array, float32
        target data
    n_iterations: int64
        total number of iterations
    N0: int64 
        number of clean data (our prior knowledge)
    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float32
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float32
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float32 
                  list of translation parameters in all interations 
    timings_list: (n_iterations,) numpy array, int 
                  list of timings at the end of all interations 
                      
    '''
    t0 = time.time()
    n,d=T.shape
    N1=S.shape[0]
    # initlize 
    rotation=np.eye(d,dtype=np.float64)    
    scalar=np.float32(1) 
    beta=vec_mean(T)-vec_mean(scalar*S.dot(rotation)) 
    #paramlist=[]
    projections=random_projections(d,n_iterations,1)
    mass_diff=0
    b=np.log((N1-N0+1)/1)
    Lambda=6*np.sum(beta**2)
    rotation_list=np.zeros((n_iterations,d,d)) #,dtype=np.float32)
    scalar_list=np.zeros((n_iterations)) #,dtype=np.float32)
    beta_list=np.zeros((n_iterations,d)) #,dtype=np.float32)
    timings_list=np.zeros((n_iterations,)) #,dtype=np.float32)
    T_hat=S.dot(rotation)*scalar+beta
    Domain_org=arange(0,N1)
    Delta=Lambda/8
    lower_bound=Lambda/10000
    for i in tqdm(range(n_iterations)):
#        print('i',i)
        theta=projections[i]
        T_hat_theta=np.dot(theta,T_hat.T)
        T_theta=np.dot(theta,T.T)
        
        T_hat_indice=T_hat_theta.argsort()
        T_indice=T_theta.argsort()
        T_hat_s=T_hat_theta[T_hat_indice]
        T_s=T_theta[T_indice]
        # c=cost_matrix(T_hat_s,T_s)
        obj,phi,psi,piRow,piCol=solve_opt(T_hat_s,T_s,Lambda)
        L=piRow.copy()
        L=recover_indice(T_hat_indice,T_indice,L)
        
      #debug 
        # if L.max()>=n:
        #     print('error')
        #     return T_hat_theta,T_theta,Lambda
        #     break
        
        #move T_hat
        Domain=Domain_org[L>=0]
        mass=Domain.shape[0]
        if Domain.shape[0]>=1:
            Range=L[L>=0]
            T_hat_take_theta=T_hat_theta[Domain]
            T_take_theta=T_theta[Range]
            T_hat[Domain]+=np.expand_dims(T_take_theta-T_hat_take_theta,1)*theta

        T_hat_take=T_hat[Domain]
        S_take=S[Domain]
        
        # compute the optimal rotation, scaling, shift
        rotation,scalar=recover_rotation(T_hat_take,S_take)
        scalar=np.sqrt(np.trace(np.cov(T_hat_take.T))/np.trace(np.cov(S_take.T)))
        beta=vec_mean(T_hat_take)-vec_mean(scalar*S_take.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta
        
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta
        timings_list[i] = time.time() - t0
        N=(N1-N0)*1/(1+b*(i/n_iterations))+N0
        mass_diff=mass-N
        if mass_diff>N*0.009:
            Lambda-=Delta 
        if mass_diff<-N*0.003:
            Lambda+=Delta
            Delta=Lambda*1/8
        if Lambda<Delta:
            Lambda=Delta
            Delta=Delta*1/2
        if Delta<lower_bound:
            Delta=lower_bound
        # if i&50==0:
        #     print('scalar',scalar)
        #     print('lambda',Lambda)
        #     print('delta',Delta)
        #     print('N',N)
        #     print('mass_diff',mass_diff)
    return rotation_list,scalar_list,beta_list, timings_list 



# our method
#@nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:]))(float64[:,:],float64[:,:],int64,int64)'])
# @nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:],float64[:]))(float64[:,:],float64[:,:],int64,int64)'],cache=True)
def pawl_main(S,T,n_iterations,N0):
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float32
        source data 
    T: (n,d) numpy array, float32
        target data
    n_iterations: int64
        total number of iterations
    N0: int64 
        number of clean data (our prior knowledge)
    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float32
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float32
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float32 
                  list of translation parameters in all interations 
    timings_list: (n_iterations,) numpy array, int 
                  list of timings at the end of all interations 
                      
    '''
    t0 = time.time()
    n,d=T.shape
    # initlize 
    rotation=np.eye(d,dtype=np.float64)    
    scalar=np.float32(1) 
    beta=vec_mean(T)-vec_mean(scalar*S.dot(rotation)) 
    #paramlist=[]
    projections=random_projections(d,n_iterations,1)
    rotation_list=np.zeros((n_iterations,d,d)) #,dtype=np.float32)
    scalar_list=np.zeros((n_iterations)) #,dtype=np.float32)
    beta_list=np.zeros((n_iterations,d)) #,dtype=np.float32)
    timings_list=np.zeros((n_iterations,)) #,dtype=np.float32)
    if N0 == "elbow":
        estimated_n0 = np.zeros((n_iterations,),dtype=np.int64)
    timings_list=np.zeros((n_iterations,)) #,dtype=np.float32)
    
    T_hat=S.dot(rotation)*scalar+beta  # T_hat := rotated S
    for i in tqdm(range(n_iterations)):
#        print('i',i)
        theta=projections[i]
        T_hat_theta=np.dot(theta,T_hat.T)  # T_hat_theta := rotated S projected on theta
        T_theta=np.dot(theta,T.T)          # T_theta     := T projected on theta
        
        if N0 == "elbow":
            indices_s, indices_t, _, elbow_index = partial_ot_1d_elbow(T_hat_theta, T_theta)
            estimated_n0[i] = elbow_index
        else:
            indices_s, indices_t, _ = partial_ot_1d(T_hat_theta, T_theta, max_iter=N0)
        
        #move T_hat
        T_hat_take_theta=T_hat_theta[indices_s]
        T_take_theta=T_theta[indices_t]
        T_hat[indices_s]+=np.expand_dims(T_take_theta-T_hat_take_theta,1)*theta

        T_hat_take=T_hat[indices_s]
        S_take=S[indices_s]
        
        # compute the optimal rotation, scaling, shift
        rotation,scalar=recover_rotation(T_hat_take,S_take)
        scalar=np.sqrt(np.trace(np.cov(T_hat_take.T))/np.trace(np.cov(S_take.T)))
        beta=vec_mean(T_hat_take)-vec_mean(scalar*S_take.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta
        
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta
        timings_list[i] = time.time() - t0
    if N0 == "elbow":
        print(estimated_n0)
    return rotation_list,scalar_list,beta_list, timings_list 


# our method
#@nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:]))(float64[:,:],float64[:,:],int64,int64,int64)'])
# @nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:],float64[:]))(float64[:,:],float64[:,:],int64,int64,int64)'],cache=True)
def swgg_main(S,T,n_iterations,N0, n_proj):
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float32
        source data 
    T: (n,d) numpy array, float32
        target data
    n_iterations: int64
        total number of iterations
    N0: int64 
        number of clean data (our prior knowledge)
    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float32
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float32
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float32 
                  list of translation parameters in all interations 
    timings_list: (n_iterations,) numpy array, int 
                  list of timings at the end of all interations 
                      
    '''
    t0 = time.time()
    n,d=T.shape
    # initlize 
    rotation=np.eye(d,dtype=np.float64)    
    scalar=np.float32(1) 
    beta=vec_mean(T)-vec_mean(scalar*S.dot(rotation)) 
    #paramlist=[]
    projections=random_projections(d,n_iterations,1)
    rotation_list=np.zeros((n_iterations,d,d)) #,dtype=np.float32)
    scalar_list=np.zeros((n_iterations)) #,dtype=np.float32)
    beta_list=np.zeros((n_iterations,d)) #,dtype=np.float32)
    timings_list=np.zeros((n_iterations,)) #,dtype=np.float32)
    if N0 == "elbow":
        estimated_n0 = np.zeros((n_iterations,),dtype=np.int64)
    timings_list=np.zeros((n_iterations,)) #,dtype=np.float32)
    
    T_hat=S.dot(rotation)*scalar+beta  # T_hat := rotated S

    pb = PartialSWGG(n_proj=n_proj, max_iter_partial=N0)

    for i in tqdm(range(n_iterations)):
#        print('i',i)
        # theta=projections[i]
        # T_hat_theta=np.dot(theta,T_hat.T)  # T_hat_theta := rotated S projected on theta
        # T_theta=np.dot(theta,T.T)          # T_theta     := T projected on theta
        
        if N0 == "elbow":
            raise NotImplementedError
            indices_s, indices_t, _, elbow_index = partial_ot_1d_elbow(T_hat_theta, T_theta)
            estimated_n0[i] = elbow_index
        else:
            projections=random_projections(d,n_proj,1)
            indices_s, indices_t, best_theta, _ = pb.fit(T_hat, T, projections)
            T_hat_theta=np.dot(best_theta,T_hat.T)  # T_hat_theta := rotated S projected on theta
            T_theta=np.dot(best_theta,T.T)          # T_theta     := T projected on theta
        
        #move T_hat
        T_hat_take_theta=T_hat_theta[indices_s]
        T_take_theta=T_theta[indices_t]
        T_hat[indices_s]+=np.expand_dims(T_take_theta-T_hat_take_theta,1)*best_theta

        T_hat_take=T_hat[indices_s]
        T_take=T[indices_t]
        S_take=S[indices_s]
        
        # compute the optimal rotation, scaling, shift
        update_along_theta_only = True
        if update_along_theta_only:
            rotation,scalar=recover_rotation(T_hat_take,S_take)
            scalar=np.sqrt(np.trace(np.cov(T_hat_take.T))/np.trace(np.cov(S_take.T)))
            beta=vec_mean(T_hat_take)-vec_mean(scalar*S_take.dot(rotation))
        else:
            rotation,scalar=recover_rotation(T_take,S_take)
            scalar=np.sqrt(np.trace(np.cov(T_take.T))/np.trace(np.cov(S_take.T)))
            beta=vec_mean(T_take)-vec_mean(scalar*S_take.dot(rotation))
        
        T_hat=S.dot(rotation)*scalar+beta
        
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta
        timings_list[i] = time.time() - t0
    if N0 == "elbow":
        print(estimated_n0)
    return rotation_list,scalar_list,beta_list, timings_list 






# method of spot_boneel 
# @nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:],float64[:]))(float64[:,:],float64[:,:],int64,int64)'],cache=True)
def spot_bonneel(S,T,n_projections=20,n_iterations=200):
    
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float64
        source data 
    T: (n,d) numpy array, float64
        target data
    n_projections: int64
        number of projections in each iteration 
    n_iterations: int64
        total number of iterations

    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float64
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float64
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float64 
                  list of translation parameters in all interations 
    timings_list: (n_iterations,) numpy array, int 
                  list of timings at the end of all interations 
                      
    '''
    t0 = time.time()
    
    n,d=T.shape
    N1=S.shape[0]
    # initlize 
    rotation=np.eye(d) #,dtype=np.float64)
    scalar=nb.float64(1.0) #
    beta=vec_mean(T)-vec_mean(scalar*S.dot(rotation))
    #paramlist=[]
    
    rotation_list=np.zeros((n_iterations,d,d)) #.astype(np.float64)
    scalar_list=np.zeros((n_iterations)) #.astype(np.float64)
    beta_list=np.zeros((n_iterations,d)) #.astype(np.float64)
    timings_list=np.zeros((n_iterations,)) #.astype(np.float64)
    T_hat=S.dot(rotation)*scalar+beta
    
    #Lx_hat_org=arange(0,n)
    
    for i in tqdm(range(n_iterations)):
#        print('i',i)

        projections=random_projections(d,n_projections,1)
        
# #        print('start1')
        T_hat=X_correspondence_pot(T_hat,T,projections)
        rotation,scalar=recover_rotation(T_hat,S)
        beta=vec_mean(T_hat)-vec_mean(scalar*S.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta

#         #move That         
        rotation_list[i]=rotation         
        scalar_list[i]=scalar
        beta_list[i]=beta
        timings_list[i] = time.time() - t0

    return rotation_list,scalar_list,beta_list, timings_list    



# @nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:],float64[:]))(float64[:,:],float64[:,:],int64)'],cache=True)
def icp_du(S,T,n_iterations):
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float64
        source data 
    T: (n,d) numpy array, float64
        target data
        
    
    n_iterations: int64
        total number of iterations

    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float64
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float64
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float64 
                  list of translation parameters in all interations 
    timings_list: (n_iterations,) numpy array, int 
                  list of timings at the end of all interations 
                      
    '''
    t0 = time.time()
    n,d=T.shape

    # initlize 
    rotation=np.eye(d) #,dtype=np.float64)
    scalar=1.0  #nb.float64(1) #
    beta=vec_mean(T)-vec_mean(scalar*np.dot(S,rotation))

    
    
    rotation_list=np.zeros((n_iterations,d,d)) #.astype(np.float64)
    scalar_list=np.zeros((n_iterations)) #.astype(np.float64)
    beta_list=np.zeros((n_iterations,d)) #.astype(np.float64)
    timings_list=np.zeros((n_iterations,)) #.astype(np.float64)
    T_hat=np.dot(S,rotation)*scalar+beta
    
    # #Lx_hat_org=arange(0,n)
    
    for i in tqdm(range(n_iterations)):
#        print('i',i)
        M=cost_matrix_d(T_hat,T)
        argmin_T=closest_y_M(M) #M.argmin(1) 
        T_take=T[argmin_T]
        T_hat=T_take
        rotation,scalar_d=recover_rotation_du(T_hat,S)
        scalar=np.mean(scalar_d)
        beta=vec_mean(T_hat)-vec_mean(scalar*S.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta
        
        #move Xhat         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta
        timings_list[i] = time.time() - t0

    return rotation_list,scalar_list,beta_list, timings_list 




# @nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:]))(float64[:,:],float64[:,:],int64)'],cache=True)
def icp_umeyama(S,T,n_iterations):
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float64
        source data 
    T: (n,d) numpy array, float64
        target data
        
    
    n_iterations: int64
        total number of iterations

    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float64
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float64
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float64 
                  list of translation parameters in all interations 
                      
    '''
        
    n,d=S.shape

    # initlize 
    rotation=np.eye(d) #,dtype=np.float64)
    scalar=1.0 #nb.float64(1.0) #
    beta=vec_mean(T)-vec_mean(scalar*S.dot(rotation))
    # paramlist=[]
    rotation_list=np.zeros((n_iterations,d,d)) #.astype(np.float64)
    scalar_list=np.zeros((n_iterations)) #.astype(np.float64)
    beta_list=np.zeros((n_iterations,d)) #.astype(np.float64)
    T_hat=S.dot(rotation)*scalar+beta
    

    
    for i in tqdm(range(n_iterations)):
#        print('i',i)
       # print(i)
        M=cost_matrix_d(T_hat,T)
        argmin_T=closest_y_M(M) #M.argmin(1) #closest_y_M(M)
        T_take=T[argmin_T]
        T_hat=T_take
        rotation,scalar=recover_rotation(T_hat,S)
        #scalar=np.mean(scalar_d)
        beta=vec_mean(T_hat)-vec_mean(scalar*S.dot(rotation))
        X_hat=S.dot(rotation)*scalar+beta
        
        #move That         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list  






# def recover_rotation_du(X,Y):
#     n,d=X.shape
#     X_c=X-torch.mean(X,0)
#     Y_c=Y-torch.mean(Y,0)
#     YX=Y_c.T@X_c
#     U,S,VT=torch.linalg.svd(YX)
#     R=U@VT
#     diag=torch.eye(d)
#     diag[d-1,d-1]=torch.det(R)
#     rotation=U@diag@VT
#     E_list=torch.eye(3)
#     scaling=torch.zeros(3)
#     for i in range(3):
#         Ei=torch.diag(E_list[i])
#         num=0
#         denum=0
#         for j in range(3):
#             num+=X_c[j].T@rotation.T@Ei@Y_c[j]
#             denum+=Y_c[j].T@Ei@Y_c[j]
#         scaling[i]=num/denum

#     return rotation,scaling

# def int_rotation(X,Y):
#     n,d=X.shape
#     X_c=X-torch.mean(X,0)
#     Y_c=Y-torch.mean(Y,0)
#     Ux,Sx,VTx=torch.linalg.svd(X_c)
#     Uy,Sy,VTy=torch.linalg.svd(Y_c)
#     R=VTy.T@VTx
#     return R
    

# def init_angle(X,Y):
#     R_es=recover_rotation(X,Y)
#     theta_es=recover_angle(R_es)
#     return theta_es


def save_parameter(rotation_list,scalar_list,beta_list,save_path):
    """
    convert parameter list and save as one file 
    
    parameter:
    ------------
    rotation_list: numpy array, shape (n_itetations, d,d), float
    scalar_list: numpy array, shape (n_itetations,), float
    beta_list: numpy array, shape (n_itetations, d), float
    save_path: string 
    """
    paramlist=[]
    N=len(rotation_list)
    for i in range(N):
        param={}
        param['rotation']=rotation_list[i]
        param['beta']=beta_list[i]
        param['scalar']=scalar_list[i]
        paramlist.append(param)
    torch.save(paramlist,save_path)
    #return paramlist

    
    

# visualization 

def get_noise_index(Y0,Y1):
    """
    get the indices of clean data and noise data of Y1 
    
    Parameters:
    ---------
    Y0: numpy array, shape (N1,d), clean data
    Y1: numpy array, shape (N1+s,d), noisy data, where s>0. Y0 \subset Y1 
    
    Returns: 
    ----------
    np.array(data_indices): numpy array, shape (N1,), int64  
    np.array(noice): numpy array, shape (s,), int64 
    """
        
    N=Y1.shape[0]
    data_indices=[]
    noise_indices=[]
    for j in range(N):
        yj=Y1[j]
        if yj in Y0:
            data_indices.append(j)
        else:
            noise_indices.append(j)
    return np.array(data_indices),np.array(noise_indices)

def shape_image(T_data,S_data,T_noise=[],S_noise=[],name=None, param=None):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    if param!=None:
        xlim,ylim,zlim,view_init,(dx,dy,dz)=param
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(view_init[0],view_init[1],vertical_axis=view_init[2])
        
    ax.scatter(T_data[:,0]+dx,T_data[:,1]+dy,T_data[:,2]+dz,alpha=.5,c='C2',s=2,marker='o')
    ax.scatter(S_data[:,0]+dx,S_data[:,1]+dy,S_data[:,2]+dz,alpha=0.5,c='C1',s=2,marker='o')
    if len(T_noise)>0:
        ax.scatter(T_noise[:,0]+dx,T_noise[:,1]+dy,T_noise[:,2]+dz,alpha=.5,c='C2',s=10,marker='o')
    if len(S_noise)>0:
        ax.scatter(S_noise[:,0]+dx,S_noise[:,1]+dy,S_noise[:,2]+dz,alpha=.5,c='C1',s=10,marker='o')
    ax.set_facecolor('black') 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(True)
    ax.axis('off')
    

    if name!=None:
        plt.savefig(name+'.pdf',dpi=200,format='pdf',bbox_inches='tight')
    else:
        plt.show()
    plt.close()

# select data 
item_list=['stanford_bunny','dragon','mumble_sitting','witchcastle']


# pamameter for ploting point cloud 
vis_param_list={'stanford_bunny':
            ([-.1,.1],[-.1,.1],[-.1,.1],(90,-90,'z'),(0.02,-0.1,0)),
'dragon':
            ([-.1,.1],[-.1,.1],[-0.1,0.1],(90,-90,'z'),(0,-0.1,0)),
'mumble_sitting':
            ([-36,36],[-36,36],[-36,36],(-20,10,'y'),(10,-10,-10)),
'witchcastle':
            ([-20,20],[-20,20],[-20,20],(45,120,'z'),(-10,20,20)),
           }    


vis_param_list0={'stanford_bunny':
            ([-.2,.2],[-.2,.2],[-.2,.2],(90,-90,'z'),(0.02,-0.1,0)),
'dragon':
            ([-.2,.2],[-.2,.2],[-0.2,0.2],(90,-90,'z'),(0,-0.1,0)),
'mumble_sitting':
            ([-66,66],[-66,66],[-66,66],(-20,10,'y'),(10,0,-10)),
'witchcastle':
            ([-38,38],[-38,38],[-38,38],(45,120,'z'),(-10,20,20)),
           }  




def init_image(X_data,X_noise,Y_data,Y_noise,image_path,name):
    """
    make a plot for the data and noise and save the plot 
    parameters: 
    X_data: numpy array, shape (n1,d), float 
        cliean data, target data 
    X_noise: numpy array, shape (s1,d), float 
        cliean data, target data 
    Y_data: numpy array, shape (n2,d), float 
        cliean data, source data 
    Y_noise: numpy array, shape (s2,d), float 
        cliean data, source data
    image_path: string 
    name: string 
        
    """
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.scatter(X_data[:,0]+5,X_data[:,1],X_data[:,2]-10,alpha=.5,c='C2',s=2,marker='o')
    ax.scatter(X_noise[:,0]+5,X_noise[:,1],X_noise[:,2]-10,alpha=.5,c='C2',s=10,marker='o')
    ax.scatter(Y_data[:,0]+5,Y_data[:,1],Y_data[:,2]-10,alpha=0.5,c='C1',s=2,marker='o')
    ax.scatter(Y_noise[:,0]+5,Y_noise[:,1],Y_noise[:,2]-10,alpha=.5,c='C1',s=10,marker='o')
    ax.set_facecolor('black') 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(True)
    ax.axis('off')
    
    # whitch_castle 
    # x+5, z-10
    ax.set_xlim([-38,38])
    ax.set_ylim([-38,38])
    ax.set_zlim([-38,38])
    ax.view_init(45,120)
    
    
    #mumble_sitting 
    # ax.set_xlim([-66,66])
    # ax.set_ylim([-66,66])
    # ax.set_zlim([-66,66])
    # ax.axis('off')
    # ax.view_init(-20,10,'y')

    
    #dragon +bunny     
    # bunny y-0.05,
    # ax.set_xlim([-.25,.25])
    # ax.set_ylim([-.25,.25])
    # ax.set_zlim([-.25,.25])
    # ax.axis('off')
    # ax.view_init( 90, -90)
    

    plt.savefig(image_path+'/'+name+'.png',dpi=200,format='png',bbox_inches='tight')
    plt.show()
    plt.close()
    
    

def normal_image(X_data,X_noise,Y_data,Y_noise,image_path,name):
    
    """
    make a plot for the data and noise and save the plot 
    truncated version 
    parameters: 
    X_data: numpy array, shape (n1,d), float 
        cliean data, target data 
    X_noise: numpy array, shape (s1,d), float 
        cliean data, target data 
    Y_data: numpy array, shape (n2,d), float 
        cliean data, source data 
    Y_noise: numpy array, shape (s2,d), float 
        cliean data, source data
    image_path: string 
    name: string 
        
    """
        
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.scatter(X_data[:,0]+3,X_data[:,1],X_data[:,2]-15,alpha=.3,c='C2',s=5,marker='o')
    ax.scatter(X_noise[:,0]+3,X_noise[:,1],X_noise[:,2]-15,alpha=.5,c='C2',s=15,marker='o')
    ax.scatter(Y_data[:,0]+3,Y_data[:,1],Y_data[:,2]-15,alpha=.9,c='C1',s=6,marker='o')
    ax.scatter(Y_noise[:,0]+3,Y_noise[:,1],Y_noise[:,2]-15,alpha=.5,c='C1',s=15,marker='o')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')
    ax.set_facecolor('black') 
    ax.grid(True)
    
    # castle,   
    #x+3, z-15 
    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])
    ax.set_zlim([-20,20])
    ax.view_init(45,120)
#    ax.view_init(0,10,'y')


    # #mumble_sitting, bunny  
    # y-10
    # ax.set_xlim([-36,36])
    # ax.set_ylim([-36,36])
    # ax.set_zlim([-36,36])
    # ax.view_init(-20,10,'y')
    #ax.view_init( 90, -90)
     
    #dragon, bunny 
    #dragon y-0.1
    #bunny, x+0.02, y-0.1    
    # ax.set_xlim([-.1,.1])
    # ax.set_ylim([-.1,.1])
    # ax.set_zlim([-.1,.1])

    # ax.view_init( 90, -90)
    # fig.set_facecolor('black')
    
    plt.savefig(image_path+'/'+name+'.png',dpi=200,format='png',bbox_inches='tight')
    plt.show()
    plt.close()
    