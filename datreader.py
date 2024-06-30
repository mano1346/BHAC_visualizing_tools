import numpy as np
import os
import f90nml

####################################################################################
########################
### Global constants ###
########################

# Size of fortran int and float
size_int=np.dtype(np.int32).itemsize
size_double=np.dtype(float).itemsize

####################################################################################
##################
### Tree class ###
##################

class tree:
    def __init__(self,ig1,ig2,ig3,level):
        self.ig=np.array([ig1,ig2,ig3],dtype=np.int32)
        self.leaf=False
        self.level=level
        self.Morton_no=0
        self.parent=None
        self.child=None
        self.neighbor={(in1,in2,in3):None for in1 in range(-1,2) for in2 in range(-1,2) for in3 in range(-1,2)}
        self.neighbor_type={(in1,in2,in3):None for in1 in range(-1,2) for in2 in range(-1,2) for in3 in range(-1,2)}
        # Neighbor types:
        # 0: the grid itself
        # 1: physical boundary
        # 2: coarser neighbor
        # 3: same resolution level
        # 4: finer neighbor

        self.next=None # Not used at the moment
        self.prev=None # Not used at the moment
    def __repr__(self):
        if self.leaf:
           return "Tree leaf ig({:d},{:d},{:d}) level {:d} ".format(self.ig[0],self.ig[1],self.ig[2],self.level)
        else:
           return "Tree node ig({:d},{:d},{:d}) level {:d} ".format(self.ig[0],self.ig[1],self.ig[2],self.level)

####################################################################################
####################
### Reader class ###
####################

class load_dat:
    """
    Loader class for dat files
    """
    def __init__(self,checkpoint,datfile='output/data',parfile='amrvac.par'):
        
        self.checkpoint=checkpoint
        self.datfile=datfile
        self.parfile=parfile
        
        self.filename="{:s}{:04d}.dat".format(self.datfile,self.checkpoint)
        
        # Initialize attributes to None
        self.mxnest=None
        self.nxlone1=None; self.nxlone2=None; self.nxlone3=None
        self.xprobmin1=None; self.xprobmin2=None; self.xprobmin3=None
        self.xprobmax1=None; self.xprobmax2=None; self.xprobmax3=None
        
        self.nleafs=None; self.levmax=None; self.ndim=None
        self.ndir=None; self.nw=None; self.nws=None
        self.neqpar=None; self.it=None; self.time=None
        
        self.eqpar=None; self.nx=None
        
        # Get info from parameter file
        par=f90nml.read(self.parfile)
        
        # Variable names (conserved variables)
        self.varlist=par['methodlist']['wnames'].split(' ')
        self.var_idx=dict([(self.varlist[i],i) for i in range(len(self.varlist))])
        # print('Variable names are:')
        # print(self.varlist)
        
        # Initializing empty dictionary for variable arrays
        self.data=dict([(self.varlist[i],None) for i in range(len(self.varlist))])

        # Grid info
        
        self.mxnest=par['amrlist']['mxnest']
        
        self.nxlone1=par['amrlist']['nxlone1']
        self.xprobmin1=par['amrlist']['xprobmin1']
        self.xprobmax1=par['amrlist']['xprobmax1']
        
        if 'nxlone2' in par['amrlist']:
            self.nxlone2=par['amrlist']['nxlone2']
            self.xprobmin2=par['amrlist']['xprobmin2']
            self.xprobmax2=par['amrlist']['xprobmax2']
        
        if 'nxlone3' in par['amrlist']:
            self.nxlone3=par['amrlist']['nxlone3']
            self.xprobmin3=par['amrlist']['xprobmin3']
            self.xprobmax3=par['amrlist']['xprobmax3'] 

        # Read footer and fill variables # 
        self.read_footer()

        # Periodic boundaris
        self.periodicB=[False,False,False]
        if par['boundlist']['typeB'][0]=='periodic':
            self.periodicB[0]=True
        if (self.ndim>1 and par['boundlist']['typeB'][2*self.nw]=='periodic'):
            self.periodicB[1]=True
        if (self.ndim>2 and par['boundlist']['typeB'][4*self.nw]=='periodic'):
            self.periodicB[2]=True
     
        # No. of blocks in the base level
        self.ng1=self.nxlone1//self.nx[0]
        self.ng2=1
        self.ng3=1
        
        if self.ndim > 1:
            self.ng2=self.nxlone2//self.nx[1]
        if self.ndim > 2:
            self.ng3=self.nxlone3//self.nx[2]
 
        # This is a dictionary containing all the trees,
        # which can be addressed by a tuple with the ig indices
        self.forest={(ig1,ig2,ig3):None for ig1 in range(self.ng1) for ig2 in range(self.ng2) for ig3 in range(self.ng3)}

        # This dictionary will contain all the nodes, addressed by ig indices and level
        self.ig_to_node={}
        # This list will contain only the leafs
        self.Morton_list=[None for i in range(self.nleafs)]
        
        # This list will contain the number of leafs per level
        #Notice that AMR level 1 in Fortran is level 0 here, etc.
        self.nleafs_level=np.zeros(self.levmax,dtype=int)
        
        self.read_forest()
                
    def read_footer(self):
        file_size=(os.stat(self.filename)).st_size
        
        # Read fixed-sized footer
        offset=file_size-size_double
        self.time=np.fromfile(self.filename, dtype=float, count= 1, sep='', offset=offset)[0]
        offset-=size_int
        self.it=np.fromfile(self.filename, dtype=np.int32, count= 1, sep='', offset=offset)[0]
        offset-=size_int
        self.neqpar=np.fromfile(self.filename, dtype=np.int32, count= 1, sep='', offset=offset)[0]
        offset-=size_int
        self.nws=np.fromfile(self.filename, dtype=np.int32, count= 1, sep='', offset=offset)[0]
        offset-=size_int
        self.nw=np.fromfile(self.filename, dtype=np.int32, count= 1, sep='', offset=offset)[0]
        offset-=size_int
        self.ndir=np.fromfile(self.filename, dtype=np.int32, count= 1, sep='', offset=offset)[0]
        offset-=size_int
        self.ndim=np.fromfile(self.filename, dtype=np.int32, count= 1, sep='', offset=offset)[0]
        offset-=size_int
        self.levmax=np.fromfile(self.filename, dtype=np.int32, count= 1, sep='', offset=offset)[0]
        offset-=size_int
        self.nleafs=np.fromfile(self.filename, dtype=np.int32, count= 1, sep='', offset=offset)[0]

        # Read variable-sized footer: physical block size and eqpar.
        offset=offset-(self.neqpar*size_double)
        self.eqpar=np.fromfile(self.filename, dtype=float, count= int(self.neqpar), sep='', offset=offset) #, *, like=None)
        offset=offset-(self.ndim*size_int)
        self.nx   =np.fromfile(self.filename, dtype=np.int32, count= int(self.ndim), sep='', offset=offset) #, *, like=None)

    ##############################
    # Methods to read the forest #
    ##############################

    def read_forest(self):
        size_block=self.nx.prod()*self.nw*size_double
        size_block_stg=(self.nx+1).prod()*self.nws*size_double
        # This is ugly, but it's the easiest way I found to modify
        # the offset and Morton number inside read_node.
        offset = [self.nleafs*(size_block+size_block_stg)]
        Morton_no=[0]
        
        level=1
        
        ng1=self.nxlone1//self.nx[0]
        ng2=1
        ng3=1
        
        if self.ndim > 1:
            ng2=self.nxlone2//self.nx[1]
        if self.ndim > 2:
            ng3=self.nxlone3//self.nx[2]
       
        # Assuming no SFC on base, otherwise this needs to change
        for ig3 in range(ng3):
            for ig2 in range(ng2):
                for ig1 in range(ng1):
                    # Add tree to forest
                    self.forest[ig1,ig2,ig3]=tree(ig1,ig2,ig3,level)
                    self.ig_to_node[ig1,ig2,ig3,1]=self.forest[ig1,ig2,ig3]
                    self.read_node(self.forest[ig1,ig2,ig3],offset,Morton_no)
                    
        # Fix connectivity
        # Go through Morton list building the connectivity
        for node in self.Morton_list:
            for idx_n3 in range(-1,2):
                for idx_n2 in range(-1,2):
                    for idx_n1 in range(-1,2):
                        self.find_neighbor_dict(node,idx_n1,idx_n2,idx_n3)
 
        # print("Forest read succesfully")
        
    #=====================================================================================    
    def read_node(self,node,offset,Morton_no):
        leaf=np.fromfile(self.filename, dtype=bool, count= 1, sep='', offset=offset[0])[0]
        offset[0]+=size_int # Fortran logicals are 4 bytes long

        # Fill node info
        node.leaf=leaf
       
        if leaf:
            self.Morton_list[Morton_no[0]]=node
            node.Morton_no=Morton_no[0]
            Morton_no[0]+=1
        else:
            child_level=node.level+1
            # Initialize children array
            node.child=[None for i in range(2**self.ndim)]

            # Loop over children
            for ch_idx in range(2**self.ndim):
                # Also this is ugly, but easiest way I found
                # to make it dimension independent
                ix1=ch_idx%2
                i1=(ch_idx-ix1)//2
                ix2=i1%2
                i2=(i1-ix2)//2
                ix3=i2%2
                
                child_ig1=2*node.ig[0]+ix1
                child_ig2=2*node.ig[1]+ix2
                child_ig3=2*node.ig[2]+ix3

                node.child[ch_idx]=tree(child_ig1,child_ig2,child_ig3,child_level)
                node.child[ch_idx].parent=node
                self.ig_to_node[child_ig1,child_ig2,child_ig3,child_level]=node.child[ch_idx]
                self.read_node(node.child[ch_idx],offset,Morton_no)

    #################################
    # Methods to build connectivity #
    #################################
    def find_neighbor_dict(self,node,idx_n1,idx_n2,idx_n3):
        # Fills neighbor pointers and information for a tree node.
        # Based on BHAC routine with same name.
        # Not considering pole structure

        if (idx_n2 != 0 and self.ndim<2) or (idx_n3 != 0 and self.ndim<3):
            return

        if idx_n1==0 and idx_n2 == 0 and idx_n3 == 0:
            # This is the block itself. Not linking to avoid infinite loops.
            node.neighbor_type[idx_n1,idx_n2,idx_n3]=0

        if node.level==1:
            self.find_root_neighbor(node,idx_n1,idx_n2,idx_n3)
            if node.neighbor[idx_n1,idx_n2,idx_n3]:
                # If neighbor exist, check if it is finer or same level
                if node.neighbor[idx_n1,idx_n2,idx_n3].leaf:
                    # Same resolution level
                    node.neighbor_type[idx_n1,idx_n2,idx_n3]=3
                else:
                    # Finer neighbor
                    node.neighbor_type[idx_n1,idx_n2,idx_n3]=4
            else:
                # If there is no neighbor, we are in a physical boundary
                node.neighbor_type[idx_n1,idx_n2,idx_n3]=1
        else:
            # Look first for coarser neighbors
            # Find child index of the block: ic
            # and neighbor indices from the parent block: inp
            ic1=node.ig[0]%2
            inp1=(ic1+idx_n1)//2
            ic2=node.ig[1]%2
            inp2=(ic2+idx_n2)//2
            ic3=node.ig[2]%2
            inp3=(ic3+idx_n3)//2

            # Grid indices of coarser neighbor 
            par_ig1=node.parent.ig[0]
            par_ig2=node.parent.ig[1]
            par_ig3=node.parent.ig[2]

            #print(par_ig1,par_ig2,par_ig3)

            par_ig_n1 = par_ig1 + inp1
            par_ig_n2 = par_ig2 + inp2
            par_ig_n3 = par_ig3 + inp3

            #print(par_ig_n1,par_ig_n2,par_ig_n3)

            # Considering periodic boundaries
            if (self.periodicB[0]):
                par_ig_n1=par_ig_n1%(self.ng1*2**(node.level-2))
            if (self.periodicB[1]):
                par_ig_n2=par_ig_n2%(self.ng2*2**(node.level-2))
            if (self.periodicB[2]):
                par_ig_n3=par_ig_n3%(self.ng3*2**(node.level-2))

            # If neighbor indices are outside the grid
            if (  par_ig_n1<0 or par_ig_n1>self.ng1*2**(node.level-2) -1
               or par_ig_n2<0 or par_ig_n2>self.ng2*2**(node.level-2) -1
               or par_ig_n3<0 or par_ig_n3>self.ng3*2**(node.level-2) -1  ):
                node.neighbor_type[idx_n1,idx_n2,idx_n3]=1
                return

            # If the coarse neighbor is a leaf
            if self.ig_to_node[par_ig_n1,par_ig_n2,par_ig_n3,node.level-1].leaf:
                node.neighbor[idx_n1,idx_n2,idx_n3] = self.ig_to_node[par_ig_n1,par_ig_n2,par_ig_n3,node.level-1]
                node.neighbor_type[idx_n1,idx_n2,idx_n3] = 2

            else:
                # Get same level indices
                n_ic1 = node.ig[0] + idx_n1
                n_ic2 = node.ig[1] + idx_n2
                n_ic3 = node.ig[2] + idx_n3
                # Considering periodic boundaries
                if (self.periodicB[0]):
                    n_ic1 = n_ic1%(self.ng1*2**(node.level-1))
                if (self.periodicB[1]):
                    n_ic2 = n_ic2%(self.ng2*2**(node.level-1))
                if (self.periodicB[2]):
                    n_ic3 = n_ic3%(self.ng3*2**(node.level-1))
                # If the same level neighbor is a leaf
                if self.ig_to_node[n_ic1,n_ic2,n_ic3,node.level].leaf:
                    node.neighbor[idx_n1,idx_n2,idx_n3] = self.ig_to_node[n_ic1,n_ic2,n_ic3,node.level]
                    node.neighbor_type[idx_n1,idx_n2,idx_n3] = 3
                # Else, the only possibility is that the neighbor is finer
                else:
                    # Assign as neighbor the parent of the finer neighbors
                    #node.neighbor[idx_n1,idx_n2,idx_n3] = self.ig_to_node[n_ic1,n_ic2,n_ic3,node.level]
                    node.neighbor_type[idx_n1,idx_n2,idx_n3] = 4

    #=====================================================================================    
    def find_root_neighbor(self,node,idx_n1,idx_n2,idx_n3):
        # Based on BHAC routine with the same name.
        # idx_n1,2,3 neighbor code: the indices of a 3x3 cube
        # where the current node has indices 0,0,0
        j1=node.ig[0]+idx_n1
        j2=node.ig[1]+idx_n2
        j3=node.ig[2]+idx_n3

        # Considering periodic boundaries
        if (self.periodicB[0]):
            j1=j1%self.ng1
        if (self.periodicB[1]):
            j2=j2%self.ng2
        if (self.periodicB[2]):
            j3=j3%self.ng3

        # If we want to consider pole information, it should be done here. Not doing now.

        if (  j1>-1 and j1<self.ng1
          and j2>-1 and j2<self.ng2
          and j3>-1 and j3<self.ng3  ):
            node.neighbor[idx_n1,idx_n2,idx_n3]=self.forest[j1,j2,j3]

        # Else do nothing, neighbors are anyways initialized to None.

    #############################
    # Methods to load variables #
    #############################
    def load_var(self,varname):
        
        if not(varname in self.varlist):
            print('Error: Variable {:s} is not in the variable list.'.format(varname))
            print('The list of variables is:')
            print(self.varlist)
            
            return
        
        # print('Loading variable {:s}'.format(varname))

        # Block size
        size_block=self.nx.prod()*self.nw*size_double
        size_block_stg=(self.nx+1).prod()*self.nws*size_double

        # Cells per block
        ncells=self.nx.prod()
                
        # Initialize arrays to zero
        var=np.zeros(self.nleafs*ncells,dtype=float)
        
        for Morton_no in range(self.nleafs):
            offset=Morton_no*(size_block+size_block_stg)+ncells*size_double*self.var_idx[varname]
            var[Morton_no*ncells:(Morton_no+1)*ncells]=np.fromfile(
                self.filename, dtype=float, count= ncells, sep='', offset=offset)
        self.data[varname]=var

    #=====================================================================================           
    def get_all(self):
        for varname in self.varlist:
            self.load_var(varname)


    ##############################
    # Methods to convert indices #
    ##############################
    def gl_idx_from_int_idx(self,Morton_no,idx1,idx2,idx3):
    # Returns global array index from Morton index and block internal indices
        if (Morton_no>self.nleafs-1 or
            idx1 > self.nx[0]-1 or
            idx2 > self.nx[1]-1 or
            idx3 > self.nx[2]-1 or
            Morton_no < 0 or
            idx1 < 0 or
            idx2 < 0 or
            idx3 < 0):
            print("Invalid input indices")
            print("Morton no.={:d}, idx1,idx2,idx3={:d},{:d},{:d}".format(Morton_no,idx1,idx2,idx3))
            print("No. of leafs={:d}, nx1,nx2,nx3={:d},{:d},{:d}".format(self.nleafs,self.nx[0],self.nx[1],self.nx[2]))
            return -1
        
        block_size_var=self.nx.prod() # Block size for a single variable
        
        return Morton_no*block_size_var+self.nx[1]*self.nx[0]*idx3+self.nx[0]*idx2+idx1

    #=====================================================================================           
    def int_idx_from_gl_idx(self,gl_idx):
     # Return Morton index and block internal indices from global array index
       
        block_size_var=self.nx.prod() # Block size for a single variable
        
        if (gl_idx > self.nleafs*block_size_var-1):
            print("Invalid global index {:d}, maximum value: {:d}".format(gl_idx,self.nleafs*block_size_var))
        
        Morton_no = gl_idx//block_size_var
        tmp_idx= gl_idx%block_size_var
        idx3=tmp_idx//(self.nx[1]*self.nx[0])
        tmp_idx=tmp_idx%(self.nx[1]*self.nx[0])
        idx2=tmp_idx//self.nx[0]
        idx1=tmp_idx%self.nx[0]
        
        return Morton_no,idx1,idx2,idx3


    def coord2flat(self,x):
        ## Gives the index of a cell in the flattened arrays that corresponds to a given position.
        ## The position is given in the internal code coordinates.
        ## If the point is outside the domain, return -1.
        ## Based on find_point_ipe in BHAC's mod_interpolate

        # Find the indices of the tree root in the forest
        ig=[0,0,0] # Block indices
        ix=[0,0,0] # Indices inside block
        xprobmin=[self.xprobmin1,self.xprobmin2,self.xprobmin3]
        xprobmax=[self.xprobmax1,self.xprobmax2,self.xprobmax3]
        ng=[self.ng1,self.ng2,self.ng3]
        dx=[1.,1.,1.]
        #nx=[self.nx1,self.nx2,self.nx3]

        # Check if the point is inside the simulation domain
        for idim in range(self.ndim):
            if (x[idim] < xprobmin[idim] or xprobmax[idim]*0.9999 < x[idim] ): return -1
            dx[idim] =  (xprobmax[idim] - xprobmin[idim])/ng[idim]
            ig[idim] = int((x[idim]-xprobmin[idim])/dx[idim])

        node=self.forest[ig[0],ig[1],ig[2]]

        # Traverse tree to the leaf that contains the point
        while (not node.leaf):
            child_idx=[0,0,0]

            for idim in range(self.ndim):
                dx[idim] *= 0.5;
                child_idx[idim] = int((x[idim]-xprobmin[idim])/dx[idim])%2

            node=node.child[child_idx[0]+2*child_idx[1]+4*child_idx[2]]

        # Find position within the block: Calculate internal cell indices
        for idim in range(self.ndim):
            # From now on, dx is the cell size instead of the block size
            dx[idim] /= self.nx[idim] 
            ix[idim] = (int((x[idim]-xprobmin[idim])/dx[idim]))%self.nx[idim]

        # Convert internal indices to index in the flattened array and return
        return self.nx[0]*self.nx[1]*self.nx[2]*node.Morton_no + self.nx[0]*self.nx[1]*ix[2] + self.nx[0]*ix[1] + ix[0];


####################################################################################

    def interpolate_at(self,x,var):
        # Interpolates the value of the variable contained in array 'var',
        # which has the same dimension as the flattened arrays in the data.

        # Find the cell that contains the point
        flat_idx = self.coord2flat(x)
        Morton_no,idx1,idx2,idx3 = self.int_idx_from_gl_idx(flat_idx)

        if Morton_no < 0:
            # print('Point outside grid, x=',x)
            return 0.0

        node = self.Morton_list[Morton_no]

        # Get cell center and cell sizes
        ig1   = node.ig[0]
        ig2   = node.ig[1]
        ig3   = node.ig[2]
        level = node.level

        dx1 = (0.5**(level-1))*(self.xprobmax1 - self.xprobmin1)/self.nxlone1
        dx2 = (0.5**(level-1))*(self.xprobmax2 - self.xprobmin2)/self.nxlone2
        dx3 = (0.5**(level-1))*(self.xprobmax3 - self.xprobmin3)/self.nxlone3

        # print('dx1,dx2,dx3',dx1,dx2,dx3)

        ## !!! There is something wrong in the formulas below. Need to check

        x1 = self.xprobmin1 + dx1*(self.ng1*(node.ig[0]-1) + idx1-1 + 0.5)
        x2 = self.xprobmin2 + dx2*(self.ng2*(node.ig[1]-1) + idx2-1 + 0.5)
        x3 = self.xprobmin3 + dx3*(self.ng3*(node.ig[2]-1) + idx3-1 + 0.5)

        # print('x1,x2,x3',x1,x2,x3)

        # Depending on the position relative to the cell center,
        # select indices ahead or behind, and get fraction relative to lowest points
        # for interpolation
        
        f1 = (x[0] - x1)/dx1
        f2 = (x[1] - x2)/dx2
        f3 = (x[2] - x3)/dx3

        # print('f1,...',f1,f2,f3)

        if x1 > x[0]:
            idx1_0 = idx1 - 1
            idx1_1 = idx1
            f1 = f1 + 1
        else:
            idx1_0 = idx1
            idx1_1 = idx1 + 1

        if x2 > x[1]:
            idx2_0 = idx2 - 1
            idx2_1 = idx2
            f2 = f2 + 1
        else:
            idx2_0 = idx2
            idx2_1 = idx2 + 1

        if x3 > x[2]:
            idx3_0 = idx3 - 1
            idx3_1 = idx3
            f3 = f3 + 1
        else:
            idx3_0 = idx3
            idx3_1 = idx3 + 1


        # If point is outside block, use flat interpolation, i.e., repeat index.
        # at the moment, this is the case even for same-resolution neighbors
        if idx1_0 < 0:
            idx1_0 = 0
        if idx1_1 > self.ng1-1:
            idx1_1 = self.ng1-1

        if idx2_0 < 0:
            idx2_0 = 0
        if idx2_1 > self.ng2-1:
            idx2_1 = self.ng2-1

        if idx3_0 < 0:
            idx3_0 = 0
        if idx3_1 > self.ng3-1:
            idx3_1 = self.ng3-1

        # Perform tri-linear interpolation
        flat_idx000 = self.gl_idx_from_int_idx(Morton_no,idx1_0,idx2_0,idx3_0)
        flat_idx100 = self.gl_idx_from_int_idx(Morton_no,idx1_1,idx2_0,idx3_0)
        flat_idx110 = self.gl_idx_from_int_idx(Morton_no,idx1_1,idx2_1,idx3_0)
        flat_idx010 = self.gl_idx_from_int_idx(Morton_no,idx1_0,idx2_1,idx3_0)
        flat_idx001 = self.gl_idx_from_int_idx(Morton_no,idx1_0,idx2_0,idx3_1)
        flat_idx101 = self.gl_idx_from_int_idx(Morton_no,idx1_1,idx2_0,idx3_1)
        flat_idx011 = self.gl_idx_from_int_idx(Morton_no,idx1_0,idx2_1,idx3_1)
        flat_idx111 = self.gl_idx_from_int_idx(Morton_no,idx1_1,idx2_1,idx3_1)


        c000 = var[flat_idx000]
        c001 = var[flat_idx001]
        c010 = var[flat_idx010]
        c011 = var[flat_idx011]
        c100 = var[flat_idx100]
        c101 = var[flat_idx101]
        c110 = var[flat_idx110]
        c111 = var[flat_idx111]

        c00 = c000*(1. - f1) + c100*f1
        c01 = c001*(1. - f1) + c101*f1
        c10 = c010*(1. - f1) + c110*f1
        c11 = c011*(1. - f1) + c111*f1

        c0 = c00*(1. - f2) + c10*f2
        c1 = c01*(1. - f2) + c11*f2

        c = c0*(1. - f3) + c1*f3

        # print('c000...',c000,c001,c010,c011,c100,c101,c110,c111)
        # print('f1,...',f1,f2,f3)

        return c

