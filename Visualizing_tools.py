import numpy as np
from multiprocessing import Pool
import asyncio
import sys
from Graphics_3D_Engine.main import GraphicsEngine
import time
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import datreader as dat

default_timer = time.time
data, var, min_val,max_val, cmap, var_func = None, None, None, None, None, None

def plane_grid_values(args):
    i, j, Ni, Nj, Dw, Dh, pos, nup, nright, nnormal = args
    offset = (i - Ni / 2) * Dw * nright + (j - Nj / 2) * Dh * nup
    ix, iy, iz = pos + offset
    r, theta, phi = convert_to_spherical(ix, iy, iz)
    flat_idx = data.coord2flat([r, theta, phi])
    return [r, theta, phi, var_func( r, theta, phi, flat_idx),0,0,0, nnormal[1], nnormal[2], nnormal[0], iy, iz, ix,]

def volume_grid_values(args):
    i, j, k, Nw, Nh, Nd, Dw, Dh, Dd, pos, nup, nright, nnormal = args

    ix = pos[0] + (i - Nw // 2) * Dw * nright[0] + (j - Nh // 2) * Dh * nup[0] + (k - Nd // 2) * Dd * nnormal[0]
    iy = pos[1] + (i - Nw // 2) * Dw * nright[1] + (j - Nh // 2) * Dh * nup[1] + (k - Nd // 2) * Dd * nnormal[1]
    iz = pos[2] + (i - Nw // 2) * Dw * nright[2] + (j - Nh // 2) * Dh * nup[2] + (k - Nd // 2) * Dd * nnormal[2]

    r, theta, phi = convert_to_spherical(ix, iy, iz)
    flat_idx = data.coord2flat([r, theta, phi])
  
    return [r, theta, phi, var_func( r, theta, phi, flat_idx),0,0,0, nnormal[1], nnormal[2], nnormal[0], iy, iz, ix,]

def sphere_grid_values(args):
    i, j, pos, radius, dtheta, dphi = args
    n_ix = np.sin(i * dtheta) * np.cos(j * dphi)
    n_iy = np.sin(i * dtheta) * np.sin(j * dphi)
    n_iz = np.cos(i * dtheta)

    ix = pos[0] + radius * n_ix
    iy = pos[1] + radius * n_iy
    iz = pos[2] + radius * n_iz

    r, theta, phi = convert_to_spherical(ix, iy, iz)
    flat_idx = data.coord2flat([r, theta, phi])
    return [r, theta, phi,var_func(r, theta, phi, flat_idx),0,0,0,n_iy,n_iz,n_ix,iy,iz,ix]


def recompute_grid(args):
    r,theta,phi = args
    flat_idx=data.coord2flat([r, theta, phi])
    return var_func( r, theta, phi, flat_idx)


def convert_to_spherical(X,Y,Z):
    R = np.sqrt(X**2 + Y**2 + Z**2)
    if R == 0:
        R = 0.0000001
    X1=np.log(R)
    X2=np.arccos(Z/R)/(2.*np.pi)
    X2=np.clip(X2,1e-3,1.-1.e-3) # Clip to avoid hitting domain limits
    X3=np.arctan2(Y,X)/(2.*np.pi) + 0.5
    X3=np.clip(X3,1e-3,1.-1.e-3) # Clip to avoid hitting domain limits
    return X1, X2, X3

def get_data(frame_idx, path_data, path_amrvac):
    return dat.load_dat(frame_idx, path_data, path_amrvac)
    

def set_var_func(func):
    global var_func
    var_func = func

def norm_var(var, min_max_var=None):
    min_val,max_val = min_max_var if min_max_var else (np.min(var), np.max(var))
    varnorm = (var - min_val) / (max_val - min_val)
    varnorm=np.maximum(0.,varnorm)
    varnorm=np.minimum(1.,varnorm)

    return varnorm

def set_var(new_data, new_var):
    global data, var
    data = new_data
    var = new_var

def set_data(new_data, new_var, min_max_var=None):
    global data, var, min_val, max_val
    data = new_data
    min_val,max_val = min_max_var if min_max_var else (np.min(new_var), np.max(new_var))
    var = (new_var - min_val) / (max_val - min_val)

def set_cmap(new_cmap):
    global cmap
    cmap = new_cmap

# def adjust_alpha_colormap(values, threshold=0.5):
#     global cmap
#     # print(max(values), min(values))
#     colors = cmap(values)
#     alpha = np.ones_like(values)  # Create an array of ones with the same shape as values
    
#     # Apply linear interpolation for alpha values below the threshold
#     mask = values <= threshold
#     alpha[mask] = (values[mask] / threshold)**2
    
#     colors[:, -1] = alpha  # Set the alpha channel of colors
#     return colors

def transferFunction(x, tf_func):
    colors = cmap(x)
    colors2 = np.array(colors)
    a = tf_func(x)
    a = np.clip(a, 0.0, 1.)
    
    colors[:, -1] =  a 
    colors[:, 0] = colors[:, 0]*a
    colors[:, 1] = colors[:, 1]*a
    colors[:, 2] = colors[:, 2]*a
    return colors, colors2

def make_colorbar(label_text, norm, tf_func):
    colors,_ = transferFunction(np.linspace(0,1,200),tf_func)
    fig, ax = plt.subplots(figsize=(2, 6))
    fig.subplots_adjust(right=0.3, left=0.2)  

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=mpl.colors.ListedColormap(colors[:,0:3]),
                                    norm=norm,
                                    orientation='vertical') 

    cb1.ax.yaxis.set_tick_params(color='white', labelsize=14)
    for tick_label in cb1.ax.get_yticklabels():
        tick_label.set_color('white')

    fig.text(0.2, 0.95 , label_text, va='center', ha='left', rotation=0, color='white', fontsize=14) 
    fig.patch.set_alpha(0)
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    rgba_buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
    image = Image.fromarray(rgba_buffer, 'RGBA')
    plt.close(fig)
    return image

class VisTool:
    def __init__(self, min_max_var=None, resolution=[100,100], depth=1):
        self.min_max_var = min_max_var
        self.resolution = resolution
        self.depth = depth 
        self.pos_scene = None
        self.scale = None
        self.grid_tri_indices = None
        self.grid =None
        self.vertex_data = None
        self.objects = []
        self.i = None
        self.object_type = None
        self.colorbar = None

    def plane_slice(self, width, height, pos, normal, up, label_cb=None, tf_func=None, pos_scene=(0,0,0), scale=(1,1,1)):
        self.object_type = "plane"
        self.tf_func = tf_func if tf_func else lambda x : 1
        self.pos_scene = pos_scene
        self.scale = scale
        self.grid_tri_indices = self.triangle_indices()
        nnormal = normal / np.linalg.norm(normal)
        nup = up - np.dot(nnormal, up) * nnormal
        nup /= np.linalg.norm(nup)
        nright = np.cross(nnormal, nup)
        nright /= np.linalg.norm(nright)

        Dw = width / self.resolution[0]
        Dh = height / self.resolution[1]

        def generate_args():
            for j in range(self.resolution[1]):
                for i in range(self.resolution[0]):
                    yield (i, j, self.resolution[0], self.resolution[1], Dw, Dh, pos, nup, nright, nnormal)

        with Pool() as pool:
            results = pool.map(plane_grid_values, generate_args())

        self.grid = np.array(results, dtype='f4')
        a = self.grid[:, 3:]
        a[:,0:4],_= transferFunction(norm_var(a[:, 0], self.min_max_var), self.tf_func)  
        self.vertex_data = [a[self.grid_tri_indices].reshape(-1,10), self.pos_scene, self.scale]
        self.colorbar = make_colorbar(label_cb,mpl.colors.Normalize(vmin=self.min_max_var[0], vmax=self.min_max_var[1]), self.tf_func)

    def sphere_slice(self,pos,radius):
        dtheta=np.pi/(self.resolution[0]-1)
        dphi=2*np.pi/(self.resolution[1]-1)

        def generate_args():
            for j in range(self.resolution[1]):
                for i in range(self.resolution[0]):
                    yield (i, j, pos, radius,dtheta,dphi, self.cmap)

        with Pool() as pool:

            results = pool.map(sphere_grid_values, generate_args())

        self.grid = np.array(results, dtype='f4')
        a = self.grid[:, 3:]
        a[:,0:4] = cmap(norm_var(a[:, 0], self.min_max_var))
        self.vertex_data = [a[self.grid_tri_indices].reshape(-1,10), self.pos_scene, self.scale]


    def volume(self, w, h, d, pos, normal, up, label_cb, tf_func, depth=100, pos_scene=(0,0,0), scale=(1,1,1)):
        self.object_type = "volume"
        self.tf_func = tf_func
        self.pos_scene = pos_scene
        self.scale = scale
        self.depth = depth
        self.grid_tri_indices = self.triangle_indices()
        # Normalize normal vector
        nnormal = normal / np.linalg.norm(normal)
        
        # Get normalized up vector for the screen (never colinear with normal)
        nup = up - np.dot(nnormal, up) * nnormal
        nup = nup / np.linalg.norm(nup)
        
        # Get normalized right vector for the screen
        nright = np.cross(nnormal, nup)
        nright = nright / np.linalg.norm(nright)
        
        # Create camera grid
        Dw = w / self.resolution[0]
        Dh = h / self.resolution[0]
        Dd = d / self.depth

        def generate_args():
            for k in range(self.depth):
                for j in range(self.resolution[0]):
                    for i in range(self.resolution[0]):
                        yield (i, j, k, self.resolution[0], self.resolution[0], self.depth, Dw, Dh, Dd, pos, nup, nright, nnormal)

        with Pool() as pool:
            results = pool.map(volume_grid_values, generate_args())

        self.grid = np.array(results, dtype='f4')
        a = self.grid[:, 3:]
        b = norm_var(a[:, 0], self.min_max_var)
        a[:,0:4], c = transferFunction(b, self.tf_func)  
        self.vertex_data = [a[self.grid_tri_indices].reshape(-1,10), self.pos_scene, self.scale]
        # sorted_indices = np.argsort(b)
        # a = a[:,0:4]
        # b = b[sorted_indices]
        # a = a[sorted_indices]
        # c = c[sorted_indices]
        # plt.plot(b,a[:,0],'--',c='r')
        # plt.plot(b,a[:,1],'--',c='g')
        # plt.plot(b,a[:,2],'--',c='b')

        # plt.plot(b,a[:,3],c='k')
        # plt.show()
        self.colorbar = make_colorbar(label_cb,mpl.colors.Normalize(vmin=self.min_max_var[0], vmax=self.min_max_var[1]), self.tf_func)




    def recompute(self):
        def generate_args():
            for i in range(self.resolution[1] * self.resolution[0] * self.depth):
                    yield (self.grid[i,0:3])

        with Pool() as pool:
            results = pool.map(recompute_grid, generate_args())

        self.grid[:, 3] = np.array(results, dtype='f4')
        a = self.grid[:, 3:]

        if self.object_type == "plane":
            a[:,0:4],_ = transferFunction(norm_var(a[:, 0], self.min_max_var), self.tf_func)  

        elif self.object_type == "volume":
            a[:,0:4],_ = transferFunction(norm_var(a[:, 0], self.min_max_var), self.tf_func) 

        self.vertex_data[0] = a[self.grid_tri_indices].reshape(-1,10)



    def triangle_indices(self):
        # Create an array of indices for a single grid
        idx = np.arange(self.resolution[0] * self.resolution[1]).reshape((self.resolution[0], self.resolution[1]))
        
        # Calculate the triangles for a single grid
        single_grid_indices = np.vstack([
            np.c_[idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()],  # First triangle in each cell
            np.c_[idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()]     # Second triangle in each cell
        ])
        
        # Repeat indices for each grid, with appropriate offsets
        if self.depth >1:
            all_indices = np.concatenate([
                single_grid_indices + offset * self.resolution[0] * self.resolution[1]
                for offset in range(self.depth)
            ])
            return all_indices
        return single_grid_indices

    def app(self, objects=[], show_screen=True, camPos=(15, 10, 15)):
        self.objects = objects
        object_data=[self.vertex_data]

        for obj in self.objects:
            object_data.append(obj.vertex_data)
        self.app = GraphicsEngine(object_data, camPos, show_screen, cb=self.colorbar)
    
    def update(self, change_data_func, i):       
        change_data_func(i) 
        self.recompute()
        for obj in self.objects:
            obj.recompute()

        # with Pool() as pool:
        #     pool.apply_async(self.recompute())
        #     for obj in self.objects:
        #         pool.apply_async(obj.recompute())
        #     pool.join()
        #     pool.close()

        object_data=[self.vertex_data]
        for obj in self.objects:
            object_data.append(obj.vertex_data)
        self.app.scene.object_data = object_data

    @staticmethod
    def _execute_task(func, args):
        func(*args)


    def save_screen(self, foldername, filename, format=".png"):
        self.app.save_image(foldername, filename, format=format)


    def plot(self, change_data_func, start_data=0, end_data=0, save_image=False, foldername="frames/", format=".png"):
        if self.app.show_screen == True:
            async def update_func():
                i=start_data
                if save_image ==True:
                    while True:
                        await asyncio.to_thread(self.update, change_data_func, i)
                        self.i = i
                        i+=1
                else:
                    while True:
                        await asyncio.to_thread(self.update, change_data_func, i)
                        i+=1

            async def render_func():
                check = True
                while check:
                    await asyncio.sleep(0.00001)
                    self.app.get_time()
                    self.app.camera.update()
                    self.app.render()
                    if self.i !=None:
                        self.save_screen(foldername,"frame{:04}".format(self.i),format)
                        print(foldername + "frame{:04}".format(i) + format)
                        self.i = None
                    check = self.app.check_events(a=False)
                    self.app.delta_time = self.app.clock.tick(60)

            async def main():
                # Create tasks for all functions
                task1 = asyncio.create_task(update_func())
                task2 = asyncio.create_task(render_func())
                # Wait for the function that checks the condition
                done, pending = await asyncio.wait([task2], return_when=asyncio.FIRST_COMPLETED)

                # Cancel the remaining tasks
                for task in pending:
                    task.cancel()

                # # Handle cancellation exceptions
                # for task in pending:
                #     try:
                #         await task
                #     except asyncio.CancelledError:
                #         print(f"{task.get_name()} was cancelled")
                sys.exit()
            # Run the main function
            asyncio.run(main())

        elif end_data != 0:
            i=start_data
            while i<end_data+start_data:
                tstart = default_timer()
                self.app.get_time()
                self.app.camera.update()
                self.update(change_data_func, i)
                self.app.render()
                self.save_screen(foldername,"frame{:04}".format(i),format)
                tend = default_timer() - tstart
                print(f'{foldername}{"frame{:04}".format(i)}{format} - Took {tend:.2f} sec')
                self.app.delta_time = self.app.clock.tick(60)
                i+=1
            self.app.clean()
        else:
            print("Need number of datasts, N_data")

    
if __name__ == "__main__":
    import datreader as dat
    import matplotlib.pyplot as plt
    import time

    cmap_names = ['plasma', 'inferno', 'magma', 'cividis', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                    'turbo', 'nipy_spectral', 'gist_ncar', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                    'turbo', 'nipy_spectral', 'gist_ncar']


    cmaps = [plt.get_cmap(name) for name in cmap_names]

    default_timer = time.time
    data = dat.load_dat(2000, 'SANE/a-15o16/output/data', 'SANE/a-15o16/amrvac.par')
    # data.load_var('d')
    # data.load_var('lfac')
    # data.load_var('b1')
    # data.load_var('b2')
    # data.load_var('b3')

    # var = np.vstack((data.data['b1'],data.data['b2'],data.data['b3'],data.data['d']/data.data['lfac']))

    # set_var(data, var)

    # def func(r, theta, phi, flat_index):
    #     b1,b2,b3, rho = var[0,flat_index], var[1,flat_index], var[2,flat_index], var[3,flat_index]
    #     B2 = (b1 * b1 * np.exp(2*r)) + (b2 * b2 * np.exp(r)**2) + (b3 * b3 * (np.exp(r) * np.sin(theta))**2)
    #     return np.log10(B2/rho)
    
    # # def tf_func(x):
    #     # a = (
    #     #     0.8*np.exp( -(x - 0.9)**2/(2*(0.1**2))) 
    #     #     +0.6*np.exp( -(x - 0.7)**2/(2*(0.03**2))) 
    #     #     + 0.3*np.exp( -(x - 0.55)**2/(2*(0.03**2)))
    #     #     + 0.3*np.exp( -(x - 0.4)**2/(2*(0.04**2)))
    #     #     +0.05*np.exp( -(x - 0.25)**2/(2*(0.08**2)))
    #     # )
    #     # return a
    # # def tf_func(x):
    # #     a = 1.*np.exp( -(x - 1.0)**2/0.001 ) + 0.7*np.exp( -(x - 0.83)**2/0.001 ) +  0.5*np.exp( -(x - 0.75)**2/0.001 ) + 0.5*np.exp( -(x - 0.58)**2/0.002 ) + 0.25*np.exp( -(x - 0.5)**2/0.001 ) + 0.1*np.exp( -(x - 0.25)**2/0.001 )
    # #     return a
    
    # def tf_func(x):
    #     a = (
    #         1.*np.exp( -(x - 1.)**2/(2*(0.04**2))) 
    #         +0.75*np.exp( -(x - 0.85)**2/(2*(0.02**2))) 
    #         +0.6*np.exp( -(x - 0.77)**2/(2*(0.02**2))) 
    #         +0.9*np.exp( -(x - 0.667)**2/(2*(0.035**2))) 
    #         + 0.1*np.exp( -(x - 0.53)**2/(2*(0.02**2))) 
    #         + 0.1*np.exp( -(x - 0.48)**2/(2*(0.02**2))) 
    #         # + 0.3*np.exp( -(x - 0.8)**2/(2*(0.1**2)))
    # #         + 0.25*np.exp( -(x - 0.4)**2/(2*(0.04**2)))
    #         # +0.2*np.exp( -(x - 0.2)**2/(2*(0.03**2)))
    #     )
    #     return a
    
    # tf_func2= None
    
    # def tf_func(x):
    #     a = (1.*np.exp( -(x - 1.0)**2/(2*(0.2**2))) 
    #         + 0.2*np.exp( -(x - 0.5)**2/(2*(0.215**2))) 
    #         + 0.5*np.exp( -(x - 0.75)**2/(2*(0.022**2))) 
    #         + 0.25*np.exp( -(x - 0.5)**2/(2*(0.022**2))) 
    #         + 0.2*np.exp( -(x - 0.25)**2/(2*(0.022**2)))
    #     )
    #     return 0.5

    data.load_var('d')
    data.load_var('lfac')
    data.load_var('xi')

    def tf_func(x):
        a = (
            1.*np.exp( -(x - 0.73)**2/(2*(0.05**2))) 
            + 0.1*np.exp( -(x - 0.4)**2/(2*(0.04**2)))     
            + 0.5*np.exp( -(x - 0.5)**2/(2*(0.02**2)))

        )
        return a
    tf_func2 = None
    rho = data.data['d']/data.data['lfac']

    p = ((data.data['xi'] / np.power(data.data['lfac'], 2)) - rho)/4

    var = np.log10(p/rho)
    set_var(data, var)

    def func(r, theta, phi, flat_index):
        return var[flat_index]

    # data.load_var('lfac')

    # def tf_func(x):
    #     a = (
    #         # 1.*np.exp( -(x - 0.8)**2/(2*(0.05**2))) 
    #         +0.6*np.exp( -(x - 0.7)**2/(2*(0.005**2))) 
    #         +0.2*np.exp( -(x - 0.74)**2/(2*(0.01**2))) 
    #     )
    #     return a
    # def tf_func2(x):
    #     a = (
    #         # 1.*np.exp( -(x - 0.8)**2/(2*(0.05**2))) 
    #         +0.6*np.exp( -(x - 0.7)**2/(2*(0.01**2))) 
    #         +0.4*np.exp( -(x - 0.7)**2/(2*(0.1**2))) 
    #     )
    #     return a

    # var = np.log10(data.data['lfac'])
    # set_var(data, var)

    # def func(r, theta, phi, flat_index):
    #     return var[flat_index]

    set_var_func(func)

    cmap = plt.get_cmap('gist_ncar')
    set_cmap(cmap)

    min_max = [-4,2]
    tstart = default_timer()
    # p1 = VisTool(resolution=[100,100], min_max_var=min_max)
    # p1.plane_slice(40, 40, [20, 20, 0], [0., 0., 1.], [1., 0., 0.], pos_scene=(-35,10,10), tf_func=tf_func2, label_cb="$log_{10}(\\frac{P}{\\rho})$", scale=(0.6,0.6,0.6))

    # p2 = VisTool(resolution=[300,300], min_max_var=min_max)
    # p2.plane_slice(40, 40, [0, 20, 20], [1., 0., 0.], [0., 1., 0.], pos_scene=(-35,10,10), tf_func=tf_func2 ,scale=(0.6,0.6,0.6))

    # p3= VisTool(resolution=[300,300], min_max_var=min_max)
    # p3.plane_slice(40, 40, [20, 0, 20], [0., 1., 0.], [0., 0., 1.], pos_scene=(-35,10,10), tf_func=tf_func2, scale=(0.6,0.6,0.6))

    p4 = VisTool(min_max_var=min_max,resolution=[100,100])
    p4.volume(250,250,250, [0, 0, 0], [1., 0.,0.], [0., 1., 0.], "$log_{10}(\\frac{B^2}{\\rho})$",tf_func, pos_scene=(-40,20,20), scale=(0.18,0.18,0.18),  depth=100) #(40,10,5)

    tend = default_timer()
    print('Total time spent on computing planes=%f sec' % (tend-tstart)) 

    camPos = (     0,       30,      59.7728 )
    p4.plot3D(camPos=camPos, show_screen=False)
    # p4.plot3D(objects=[p2,p3,p1], camPos=camPos, show_screen=False)
    # p1.plot3D(objects=[p2,p3],camPos=camPos, show_screen=False)



    def change_data(i):
        # data = dat.load_dat(3510, 'MAD/output/data', 'MAD/amrvac.par')
        # if i ==2:
        data = dat.load_dat(2000, 'SANE/a-15o16/output/data', 'SANE/a-15o16/amrvac.par')
        data.load_var('d')
        # data.load_var('lfac')
        # data.load_var('b1')
        # data.load_var('b2')
        # data.load_var('b3')
        # var = np.vstack((data.data['b1'],data.data['b2'],data.data['b3'],data.data['d']/data.data['lfac']))
        # set_var(data, var)
        data.load_var('lfac')
        data.load_var('xi')

        rho = data.data['d']/data.data['lfac']

        p = ((data.data['xi'] / np.power(data.data['lfac'], 2)) - rho)/4

        var = np.log10(p/rho)
        set_var(data,var)


    tstart = default_timer()
    print("start plot")
    p4.plot(change_data, N_data=1, save_image=True, foldername="frames/", format=".png")
    tend = default_timer()
    print('Total time spent on rendering planes=%f sec' % (tend-tstart)) 
