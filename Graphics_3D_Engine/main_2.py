import pygame as pg
import moderngl as mgl

import sys
from .model import *
from .camera import Camera
from .light import Light
from .mesh import Mesh
from .scene import Scene
from PIL import Image

class GraphicsEngine:
    def __init__(self, object_data, win_size=(1600, 900)):
        # init pygame modules
        pg.init()
        # window size
        self.WIN_SIZE = win_size
        # set opengl attr
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        # create opengl context
        # self.screen = pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
        # mouse settings
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)
        # detect and use existing opengl context
        
        self.ctx = mgl.create_standalone_context()
        # self.ctx.front_face = 'cw'
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)
        self.ctx.enable(mgl.BLEND)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture(win_size, 4)])
        self.fbo.use()
        # self.ctx.screen.use()
        # create an object to help track time
        self.clock = pg.time.Clock()
        self.time = 0
        self.delta_time = 0
        # light
        self.light = Light()
        # camera
        self.camera = Camera(self)
        # mesh
        self.mesh = Mesh(self)
        # scene
        self.scene = Scene(self, object_data)

    def check_events(self, a=True):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.mesh.destroy()
                pg.quit()
                if a == True:
                    sys.exit()
                return False
        return True
    
    def clean(self):
        self.mesh.destroy()
        pg.quit()

    def render(self):
        # clear framebuffer
        self.ctx.clear(color=(0.08, 0.16, 0.18))
        # render scene
        self.scene.render()
        # swap buffers
        # pg.display.flip()

    def get_time(self):
        self.time = pg.time.get_ticks() * 0.001
        
    def save_image(self,foldername, filename, format=".png"):
        self.ctx.finish()
        pixels = self.fbo.read(components=3)
        image = Image.frombytes('RGB', self.WIN_SIZE, pixels)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL's origin is at the bottom-left corner
        image.save(foldername + filename + format)



if __name__ == '__main__':
    import numpy as np

    # @staticmethod
    # def get_data(vertices, indices):
    #     data = [vertices[ind] for triangle in indices for ind in triangle]
    #     return np.array(data, dtype='f4')

    # def get_vertex_data(u,v, g):
    #     vertices = [(-1, -1, 1), ( 1, -1,  1), (1,  1,  1), (-1, 1,  1),
    #                 (-1, 1, -1), (-1, -1, -1), (1, -1, -1), ( 1, 1, -1)]

    #     indices = [(0, 2, 3), (0, 1, 2),
    #                (1, 7, 2), (1, 6, 7),
    #                (6, 5, 4), (4, 7, 6),
    #                (3, 4, 5), (3, 5, 0),
    #                (3, 7, 4), (3, 2, 7),
    #                (0, 6, 1), (0, 5, 6)]
    #     vertex_data = get_data(vertices, indices)

    #     tex_coord_vertices = [(u, u, g), (v, u, g), (v, v, g), (u, v, g)]
    #     tex_coord_indices = [(0, 2, 3), (0, 1, 2),
    #                          (0, 2, 3), (0, 1, 2),
    #                          (0, 1, 2), (2, 3, 0),
    #                          (2, 3, 0), (2, 0, 1),
    #                          (0, 2, 3), (0, 1, 2),
    #                          (3, 1, 2), (3, 0, 1),]
    #     tex_coord_data = get_data(tex_coord_vertices, tex_coord_indices)

    #     normals = [( 0, 0, 1) * 6,
    #                ( 1, 0, 0) * 6,
    #                ( 0, 0,-1) * 6,
    #                (-1, 0, 0) * 6,
    #                ( 0, 1, 0) * 6,
    #                ( 0,-1, 0) * 6,]
    #     normals = np.array(normals, dtype='f4').reshape(36, 3)

    #     vertex_data = np.hstack([normals, vertex_data])
    #     vertex_data = np.hstack([tex_coord_data, vertex_data])
    #     return vertex_data

    # vertex_data = get_vertex_data(0,1, 0.5)
    from ..slicing_tools2 import Slice, set_data, set_cmap
    import datreader as dat
    import matplotlib.pyplot as plt
    import time


    default_timer = time.time
    data = dat.load_dat(2000, 'SANE/a-15o16/output/data', 'SANE/a-15o16/amrvac.par')
    data.load_var('d')

    cmap = plt.get_cmap('viridis')

    var = np.log10(data.data['d'])
    p1 = Slice(data,var,cmap)
    p1.plane_slice(8, 8, [4, 4, 0], [0., 0., 1.], [1., 0., 0.])

    p2 = Slice(data,var,cmap)
    p2.plane_slice(8, 8, [0, 4, 4], [1., 0., 0.], [0., 1., 0.])

    p3 = Slice(data,var,cmap)
    p3.plane_slice(8, 8, [4, 0, 4], [0., 1., 0.], [0., 0., 1.])


    app = GraphicsEngine([p1.vertex_data,p2.vertex_data, p3.vertex_data])
    # app = GraphicsEngine([p1.vertex_data])
    # while True:
    #     app.get_time()
    #     app.check_events()
    #     app.camera.update()
    #     app.scene.object_data  = [p1.vertex_data,p2.vertex_data, p3.vertex_data]
    #     app.render()
    #     app.delta_time = app.clock.tick(60)
    #     # print(app.scene.vertex_data[0])
    cmap_names = ['plasma', 'inferno', 'magma', 'cividis', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                    'turbo', 'nipy_spectral', 'gist_ncar', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                    'turbo', 'nipy_spectral', 'gist_ncar']
    cmaps = [plt.get_cmap(name) for name in cmap_names]

    
    def change_data(cmap, i):
        # data = dat.load_dat(2000, 'SANE/a-15o16/output/data', 'SANE/a-15o16/amrvac.par')
        # data.load_var('d')
        # var = np.log10(data.data['d'])
        # set_data(data, var)
        start = default_timer()
        set_cmap(cmap)
        p1.recomp()
        p2.recomp()
        p3.recomp()
        app.scene.object_data  = [p1.vertex_data,p2.vertex_data, p3.vertex_data]
        # app.scene.object_data  = [p1.vertex_data]
        end_rend = default_timer()
        print('computing time: %f s' % (end_rend-start)) 
    
        
    import asyncio
    # pause_event = asyncio.Event()
    # pause_event.set()

    async def long_running_function():
        i = 0
        while True:
            await asyncio.to_thread(change_data, cmaps[i], i)
            # pause_event.clear()
            # await asyncio.to_thread(app.save_image, i)
            # pause_event.set()
            i +=1

    async def other_function_1():
        check = True
        while check:
            # await pause_event.wait() 
            await asyncio.sleep(0.0001)
            app.get_time()
            app.camera.update()
            # app.scene.object_data  = [p1.vertex_data,p2.vertex_data, p3.vertex_data]
            app.render()
            check = app.check_events(a=False)
            app.delta_time = app.clock.tick(60)


    async def main():
        # Create tasks for all functions
        task1 = asyncio.create_task(long_running_function())
        task2 = asyncio.create_task(other_function_1())
        # Wait for the function that checks the condition
        done, pending = await asyncio.wait([task2], return_when=asyncio.FIRST_COMPLETED)

        # Cancel the remaining tasks
        for task in pending:
            task.cancel()

        # Handle cancellation exceptions
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                print(f"{task.get_name()} was cancelled")

        # Optionally wait for the completed task and process the result
        # for task in done:
        #     result = await task
        #     print(result)

        sys.exit()
    # Run the main function
    asyncio.run(main())


    




























