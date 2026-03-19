import numpy as np

class Grid():
    def __init__(self):
        
        self.mm = 20 # Rows
        self.nn = 20 # Columns
        self.TAO = 0.9 # Memory
        self.II = 0.3 #Exitasion strength related to the neighbour
        self.SIGMA = 0.24 # width of the gaussian distrubution 
        self.SIGMA2 = self.SIGMA**2
        self.TT = 0.05 #inhibition thereshold, to filter out some of the noice
        self.grid_gain = [0.04,0.05,0.06,0.07,0.08] # grid cells does not form only one grid map, there are multiple at different scales
        self.grid_layers = len(self.grid_gain)  
        self.grid_activity = np.random.uniform(0,1,(self.mm,self.nn,self.grid_layers))  
        self.distTri = self.buildTopology(self.mm,self.nn) #builds the twisted torus explanation below. 


    def update(self, speedVector):
        """"
        It runs once every time step. It takes the robot's current speed, updates the physics of the neural network, 
        and produces the final grid pattern.
        """
        

        self.speedVector = speedVector
        
        grid_ActTemp = []
        for jj in range(0,self.grid_layers):
            rrr = self.grid_gain[jj]*np.exp(1j*0)
            matWeights = self.updateWeight(self.distTri,rrr)
            activityVect = np.ravel(self.grid_activity[:,:,jj])
            activityVect = self.Bfunc(activityVect, matWeights)
            activityTemp = activityVect.reshape(self.mm,self.nn)
            activityTemp += self.TAO *( activityTemp/np.mean(activityTemp) - activityTemp)
            activityTemp[activityTemp<0] = 0

            self.grid_activity[:,:,jj] = (activityTemp-np.min(activityTemp))/(  np.max(activityTemp)-np.min(activityTemp)) * 30  ##Eq 2
                        

    def buildTopology(self,mm,nn):  
        """
        Build connectivity matrix     ### Eq 4
        Creates the arcitecture of the CAN grid cell network. Calvualting the distance between every pair of neurons.
        Importantly it is a twisted torous or a donut shape that is created, so it does not have edges witch is important
        So that when an agent "goes over" the edge he reapers in a way. 
        
        """
        
        #normalize the neuron indices to a scale of roughly 0 to 1.
        mmm = (np.arange(mm)+(0.5/mm))/mm
        #normalize the neuron indices to a scale of roughly 0 to 1.
        # np.sqrt(3)/2 - forces into a hexagonal grid. Which is bad for my project
        nnn = ((np.arange(nn)+(0.5/nn))/nn)*np.sqrt(3)/2
        # creates the grid. Giving each neuron a position
        xx,yy = np.meshgrid(mmm, nnn)
        #creates a colpnex number, to be able to use the euclidian norm 
        posv = xx+1j * yy
        # connectes the edges
        Sdist = [ 0+1j*0, -0.5+1j*np.sqrt(3)/2, -0.5+1j*(-np.sqrt(3)/2), 0.5+1j*np.sqrt(3)/2, 0.5+1j*(-np.sqrt(3)/2), -1+1j*0, 1+1j*0]      
        #the rest of the code below calculates the shortest distance, from neuron A to B. Or walking of the edge and 
        #get up on the other side and then go to B
        xx,yy = np.meshgrid( np.ravel(posv) , np.ravel(posv) )
        distmat = xx-yy
        for ii in range(len(Sdist)):
            aaa1 = abs(distmat)
            rrr = xx-yy + Sdist[ii]
            aaa2 = abs(rrr)
            iii = np.where(aaa2<aaa1)
            distmat[iii] = rrr[iii]
        return distmat.transpose()

    def updateWeight(self,topology,rrr): # Slight update on weights based on speed vector.
        """
        Eq 3, 
        Updates the connection weights between neurons based on the current speed of the robot
        description page 15 in the Grid model for deeper descirption 
        
        Is how the CAN network bump activity actually is updated. 
        
        topology is the distance to a neighbor 
        rrr is the scaling factor
        """
        matWeights = self.II * np.exp((-abs(topology-rrr*self.speedVector)**2)/self.SIGMA2) - self.TT   ## Eq 3
        return matWeights
     
     
    def Bfunc(self,activity, matWeights):  
        """
        Eq 1, this is represents the synaptic drive (total exitatory or inhibitory input a neuron recives). 
        A grid cell´s activity is updated based on the weighted sum of the neighbours in the network.
        
        np.dot(activity,matWeights) calculates the dot product between the current activity vector (how the cells are firing) 
        and the weight matrix matWeights (connections between cells). 
        """
        
        activity += np.dot(activity,matWeights)
        return activity