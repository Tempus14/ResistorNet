#!/usr/bin/env python
# coding: utf-8

# # Combination of all important functions
# This is a summary of the important functions for constructing a network and solving it for the currents.   
# First off there is the Construct function which takes the paths to the input data and builds a corresponding Networkx-Graph from it.  
# Then there ist the Solve class which takes a network as input and can then construct the matrices and vectors needed to solve for the currents.   
# In the end there is a snippet attached which can be used to orient the edges/current-directions. It takes a incidence-like matrix as input and changes the signs according to the potential landscape.

# *Stand 17.01.24*


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import sparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.spatial.distance import euclidean
import pandas as pd
from decimal import Decimal

WDir="/home/oliver/notebooks/Master-Arbeit/ValidationWithMedDMR/medDMR/"

def Gauss(x, mu, sig, A):
    return A  *np.exp(-0.5 * ((x-mu)/sig)**2) * 1 /(np.sqrt(2 * np.pi * sig **2))

def Load(fname, Split=True):
    a=(np.genfromtxt(fname, delimiter =" ")).flatten()    
    if Split==True:
        #Lightforge uses an high but finite value (approx 300) to handle hopping onto already taken sites
        low=a[np.abs(a)<300]
        return low
    else:
        return a

def FitGauss(data):
    bin_heights, bin_borders, _ = plt.hist(data, bins='auto')
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    popt, _ = curve_fit(Gauss, bin_centers, bin_heights, p0=[0.01, 0.009, len(data)/5])
    plt.close()
    return popt, bin_borders


def miller_abrahams_rate(J, delta_E, T_eff):
    h_bar = 1.05457173e-34  # Si
    delta_E = (delta_E + np.absolute(delta_E)) / 2.0
    miller_prefactor = np.pi / (h_bar * 2 * T_eff)
    rate = miller_prefactor * J * np.exp(-delta_E / T_eff)
    return rate
    

def miller_j(j0,a0,distances,r0=0.0):
    #Note the square to match lightforge implementation
    R = distances
    j_r0 = np.exp(-4*a0*r0)
    return (j0**2* np.clip(np.exp(-4*a0*R), a_min=0.0, a_max=j_r0))    
    
#Usage:
#Gitter=Construct()
def Construct(Coo=WDir+"results/material/device_data/coord_0.dat",#
              Typ=WDir+"results/material/device_data/mol_types_0.dat",#
              Ener=WDir+"results/material/device_data/site_energies_0.dat",#
              CoulombHoles=WDir+"rates/1_100000_holes_dE_coulomb.dat",#
              CoulombElectrons=WDir+"rates/1_100000_electrons_dE_coulomb.dat",#
              ChargeDist=WDir+"/results/experiments/particle_densities/all_data_points/site_charge_density_0.npz",#
              InjectionType="h",# for electrode simulation there are otherwise HOMO and LUMO in the same file
              J0=0.001, decay=0.1, T=300, Lam=0.2):
    #from settings file
    #J = attempt frequency (largest possible rate) =>  "maximum ti"
    #Lam =  Reorganization Energy = lambda = materials => l
    k_boltzmann = 1.3806488e-23  # Si   # 0.00008617 in eV
    kbT=k_boltzmann*T #Si
    H_Bar = 1.05457173e-34  # Si
    Qe = 1.602176634*(10**(-19)) # Electron charge
    J0*=Qe #To match lightforge implementation
    
    #Read coordinates from file
    coords = list(map(tuple, np.genfromtxt(Coo, delimiter=" ")))
    
    #Build Graph
    F=nx.Graph()
    F.add_nodes_from(coords)
    #Read Attributes
    types = np.genfromtxt(Typ, delimiter=" ")

    #The next part is due to the way lightforge exports the values into the same file
    if InjectionType=="h":
        energies = np.genfromtxt(Ener, delimiter=" ")[:,0]
        VZ=1
        #Read occupation probabilites from file
        occ=np.load(ChargeDist)["hole_density"]
    elif InjectionType=="e":
        energies = np.genfromtxt(Ener, delimiter=" ")[:,1]
        VZ=-1
        #Read occupation probabilites from file
        occ=np.load(ChargeDist)["electron_density"]
    else:
        energies = np.genfromtxt(Ener, delimiter=" ")
        VZ=1
        #Read occupation probabilites from file
        occ=np.load(ChargeDist)["hole_density"]

    #Set Attributes
    i=0
    for u in F.nodes():
        F.nodes[u]["pos"]=u
        F.nodes[u]["type"]=types[i]
        F.nodes[u]["energies"]=energies[i]
        F.nodes[u]["potential"]=energies[i]#/Qe #I guess doesn't need to be stored - can be calculated on the fly during current calculation
        F.nodes[u]["occ"]=occ[i]
        i+=1  
    #Construct Edges
    F.add_edges_from(nx.geometric_edges(F, 1.9))
    
    #Calculate Distribution for Coulombenergydifferences
    ele=Load(CoulombElectrons)
    hol=Load(CoulombHoles)
    data=np.concatenate((ele, hol))
    popt, bin_borders=FitGauss(data)
    

    
    ## Inizializing Miller Rates ##
    for (u, v) in F.edges():     
        
        ## v is start and u is end and vice versa
        ## maybe neg. VZ Prefactor to favor energetical down/up hill hopping of electrons/holes
        
        #At the moment there is no explicit implementation of the coulomb interation - except for being encoded into the occupation probabilites
        deltaE=(F.nodes[u]["energies"] - F.nodes[v]["energies"]) #+ np.random.default_rng().normal(popt[0], popt[1]) 
        deltaE*= 1.60218e-19 #Si
        J=miller_j(J0,decay,(euclidean(u,v)), 1.0)

        # proper implementation of occupation probabilites
        rateVU=miller_abrahams_rate(J, deltaE , kbT)*F.nodes[v]["occ"]*(1-F.nodes[u]["occ"])
        rateUV=miller_abrahams_rate(J, -deltaE , kbT)*F.nodes[u]["occ"]*(1-F.nodes[v]["occ"])

        conduct = (Qe**2 * (rateVU-rateUV) ) /deltaE #effective current due to occupation probability and backhopping
        F.edges[u,v]['weight'] = conduct
        
        #The following maybe has to be reworked - it is sufficient as is if one only wants currents along the x-axis
        #as the nodelist is sorted by along this axis and therefore the currentdirection should match
        #Otherwise the edge has to be oriented according to the effective current
        
#         if VZ*deltaE >0: ## electron=> flow from high to low => energetically favorable (holes the other way around) 
#             F.edges[u,v]['weight'] = resistance
#             F.edges[u,v]['prefactor'] = 1                  # => right orientation
#         else: ## reorient edge
#             # Could not just delete edge and add oposite as (Non-Di-)Graphs sort Edges by the sequence of nodes
#             # Therefore I just add a negative sign to the resistor and thereby flipping the current direction
#             F.edges[u,v]['weight'] = (-1) * resistance
#             F.edges[u,v]['prefactor'] = -1
    
    return F




#Usage:
#Solve(Gitter).currents()   
class Solve:
    def __init__(self, X):
        self.G=X
        self.N=self.G.order()
        self.nR=self.G.number_of_edges()
        
    def conductances(self):
        #Extract the values of the resistors from the graph and build a nR x nR matrix
        mat=sparse.spdiags(np.asarray(list((nx.get_edge_attributes(self.G, "weight").values()))), 0, self.nR, self.nR)
        return sparse.csc_matrix(mat)
    
    def incidence(self):
        #Builds the incidence matrix from the graph
        mat= np.transpose(nx.incidence_matrix(self.G, oriented=1)) #Beachte Transpose damit die Dimensionen der networkx funktion zum paper passen
        
        
        return mat
    
    def voltages(self):
        #Get the potential values from the nodes and build a vector
        vec=np.array(list(nx.get_node_attributes(self.G, "potential").values()))
        return vec

    
    def currents(self):
        #Combines the other functions to get the currents trough the resistors
        return (self.conductances() @ self.incidence()) @ self.voltages()

    ## The following are helper functions to get a layerwise current
    #Note this is in no way optimized for calculation of currents through multiple layers
    def total_layer(self, layer):
        lowXEnd=np.array(self.G.edges())[:,:,0]==np.min(np.array(self.G.nodes())[:,0])+layer
        lowFilter=np.logical_xor(lowXEnd[:,0],lowXEnd[:,1])
        total_IN=np.sum(self.currents()[lowFilter])
        return total_IN
    
    def total_layer_Y(self, layer):
        lowXEnd=np.array(self.G.edges())[:,:,1]==np.min(np.array(self.G.nodes())[:,1])+layer
        lowFilter=np.logical_xor(lowXEnd[:,0],lowXEnd[:,1])
        total_IN=np.sum(self.currents()[lowFilter])
        return total_IN
    
    
    def total_in(self):
        lowXEnd=np.array(self.G.edges())[:,:,0]==np.min(np.array(self.G.nodes())[:,0])
        lowFilter=np.logical_xor(lowXEnd[:,0],lowXEnd[:,1])
        total_IN=np.sum(self.currents()[lowFilter])
        return total_IN
    
    def total_out(self):
        highXEnd=np.array(self.G.edges())[:,:,0]==np.max(np.array(self.G.nodes())[:,0])
        highFilter=np.logical_xor(highXEnd[:,0],highXEnd[:,1])
        total_OUT=np.sum(self.currents()[highFilter])
        return total_OUT

 #Make values more readable   
def Format(value):
    return f"{Decimal(value):.2E}"


'''
# ##### This is a minimal working example:
Gitter=Construct()
Setup=Solve(Gitter)
#Currents=Setup.currents()
total_OUT=Setup.total_out()
total_IN=Setup.total_in()
print("IN")
print(total_IN)
print("OUT")
print(total_OUT)
print("Output in mA/cm^2") #for 20x20 grid
print(Format(float(total_OUT)/((20*10**-9)**2) * ((10**-2)**2) *1000)) #Unit conversion to mA/cm**2
# #### Mind that I used a 20x20 grid for the unit conversion at the end - this has to be adapted fot other geometries!
'''
