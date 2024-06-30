from .vao import VAO
import numpy as np

class Mesh:
    def __init__(self, app):
        self.app = app
        self.vao = VAO(app.ctx, np.array([[0.,  0.,  0.,  0.,  0., 0., 0., 0.],[0.,  0.,  0.,  0.,  0., 0., 0., 0.],[0.,  0.,  0.,  0.,  0., 0., 0., 0.]],dtype='f4'))

    def destroy(self):
        self.vao.destroy()