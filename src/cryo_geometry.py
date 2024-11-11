import odl
from odl.contrib import torch as odl_torch
import numpy as np
import torch.nn as nn


class Initialization:
    def __init__(self, args, offset_angles=False):
        
        self.img_size = args.img_size
        self.prj_size = args.prj_size
        self.n_prj = args.n_prj
        self.reso = 0.1 # physical resolution for each voxel

        rot_axis = 1 # rotate about y-axis
        if rot_axis == 0:
            self.rot_axis = [1, 0, 0]
        elif rot_axis == 1:
            self.rot_axis = [0, 1, 0]
        elif rot_axis == 2:
            self.rot_axis = [0, 0, 1]
        else:
            raise NotImplementedError

        # imaging object (recon objective) w object center as origin
        self.nx, self.ny, self.nz = self.img_size
        self.sx, self.sy, self.sz = (ii * self.reso for ii in self.img_size)

        # detector
        self.sh, self.sw = self.sx, self.sy
        self.nh, self.nw = self.prj_size # note: sinogram shape prj_size**2

        if args.fn_tlt: # if angles pre-defined in a file
            with open(args.fn_tlt, 'r') as f:
                self.angle_lst = [float(line.strip()) for line in f]
            self.angle_lst = np.array(self.angle_lst)
        else: # set angles as [-60, +60] w two-degree angular resolution
            angle_min = -60
            angle_max = 60
            angle_res = 2
            self.angle_lst = np.array(range(angle_min,
                                            angle_max + angle_res,
                                            angle_res))
        
        if offset_angles: # offset angles by 1 degree. for testing purposes only
            print('offset angles by 1 degree')
            self.angle_lst -= 1
       
        print(f'angles min: {min(self.angle_lst)}, max: {max(self.angle_lst)} degrees')
        
        self.angle_lst = np.deg2rad(self.angle_lst) # convert degrees --> radians


def build_geometry(param):
    ''' build uniformly discretized reconstruction space for imaging object
            reconstruction space: discretized functions on the cube
            e.g. [-20, 20]^3 with 300 samples per dimension. 

        returns
            recon_space: recon space for imaging object
            ray_trafo: RayTransform operator
            fbp_trafo: filtered back projection operator, approx. inverse(ray_trafo)
     '''

    # min/max corners of desired function domain
    domain_max = [param.sx / 2., param.sy / 2., param.sz / 2.]
    domain_min = [-ii for ii in domain_max]
    shape = [param.nx, param.ny, param.nz] # number of samples per axis

    # create a uniformly discretized L^p function space
    recon_space = odl.uniform_discr(min_pt=domain_min, max_pt=domain_max,
                                   shape=shape, dtype='float32')
    
    angle_partition = odl.nonuniform_partition(param.angle_lst)

    # partition with equally sized cells for detector line/plane
    # detector: uniformly sampled, n = (512, 512), min = (-30, -30), max = (30, 30)
    max_pt = [param.sh / 2., param.sw / 2.] # upper limit of interval in box
    min_pt = [-ii for ii in max_pt] # lower limit of interval in box
    shape = [param.nh, param.nw] # number of nodes
    detector_partition = odl.uniform_partition(min_pt=min_pt, max_pt=max_pt,
                                               shape=shape)

    ''' Parallel3dAxisGeometry: parallel-beam geometry for 3D-2D projection
        params:
            apart: partition of the angle interval
            dpart: partition of the detector parameter interval
            axis: rotation axis is z-axis (0,0,1),
                  initial source-to-detector direction is (0,1,0)
                  default detector axes is [(1,0,0), (0,0,1)] '''
    geometry = odl.tomo.Parallel3dAxisGeometry(apart=angle_partition,
                                               dpart=detector_partition, 
                                               axis=[0, 1, 0])
    
    ''' RayTransform: linear radon transform operator b/w L^p spaces
        params:
           vol_space: discretized recon space, i.e. domain of forward projector
           geometry: geometry of transform
           impl: implementation back-end '''
    ray_trafo = odl.tomo.RayTransform(vol_space=recon_space, 
                                      geometry=geometry, 
                                      impl='astra_cuda') 
    
    # filtered back-projection operator by approximating the inverse RayTransform 
    fbp_trafo = odl.tomo.fbp_op(ray_trafo=ray_trafo, 
                                filter_type='Ram-Lak', # most noise senstive filter
                                frequency_scaling=1.0) # relative filter cutoff freq 
    
    return recon_space, ray_trafo, fbp_trafo 


# Projector
class Projection_3DParallel(nn.Module):
    def __init__(self, param):
        super(Projection_3DParallel, self).__init__()
        
        self.param = param
        self.reso = param.reso
        
        # RayTransform operator
        _, ray_trafo, _ = build_geometry(self.param)
        
        # Wrap pytorch module
        self.trafo = odl_torch.OperatorModule(ray_trafo)

        self.back_projector = odl_torch.OperatorModule(ray_trafo.adjoint)

    def forward(self, x):
        ## x: B*C*nProj*nSino
        # note: applying projector and FBP operator should render original image
        x = self.trafo(x)
        x = x / self.reso # scale projections according to pixel resolution
        return x
    
    def back_projection(self, x):
        x = x * self.reso
        x = self.back_projector(x)
        return x


# FBP reconstruction
class FBP_3DParallel(nn.Module):
    def __init__(self, param):
        super(FBP_3DParallel, self).__init__()
        
        self.param = param
        self.reso = param.reso
        
        # filtered back-projection operator = filter operator + back-projection operator
        recon_space, ray_trafo, fbp_trafo = build_geometry(self.param)
        
        # wrap pytorch module
        self.fbp = odl_torch.OperatorModule(fbp_trafo)

        self.bp = odl_torch.OperatorModule(ray_trafo.adjoint)
        
        # filter operator for FBP (could only construct the filter only)
        filter = odl.tomo.fbp_filter_op(ray_trafo=ray_trafo, 
                                        filter_type='Ram-Lak', 
                                        frequency_scaling=1.0)
        self.filter = odl_torch.OperatorModule(filter)

    def forward(self, x):
        x = x * self.reso
        out = self.fbp(x)

        return out


class Parallel3DProjector():
    def __init__(self, args, offset_angles=False):

        # initialize required parameters for image, view, detector
        geo_param = Initialization(args=args, 
                                   offset_angles=offset_angles,
                                  )

        # forward projection function
        self.forward_projector = Projection_3DParallel(geo_param)

        # filtered back-projection
        self.filtered_back_projector = FBP_3DParallel(geo_param)
    
    def forward_project(self, volume, args):
        ''' volume: tensor w size (N, img_x, img_y, img_z) '''      
        
        # input size: (B, C, img_x, img_y, img_z)
        proj_data = self.forward_projector(volume)
        
        return proj_data

    def backward_project(self, projs):
        ''' perform filtered back-projection
            projs: tensor w size (N, n_prj, prj_size_h, prj_size_w) '''

        volume = self.filtered_back_projector(projs)  # (B, C, img_x, img_y, img_z)

        return volume
