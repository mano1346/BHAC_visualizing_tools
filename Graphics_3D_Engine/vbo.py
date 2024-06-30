
class VBO:
    def __init__(self, ctx, vertex_data):
        self.vbos = {}
        self.vbos['triangles'] = TrianglesVBO(ctx, vertex_data)

    def destroy(self):
        [vbo.destroy() for vbo in self.vbos.values()]


class BaseVBO:
    def __init__(self, ctx):
        self.ctx = ctx
        self.vbo = self.get_vbo()
        self.format: str = None
        self.attribs: list = None

    def get_vertex_data(self): ...

    def get_vbo(self):
        vbo = self.ctx.buffer(self.vertex_array)
        return vbo

    def destroy(self):
        self.vbo.release()


class TrianglesVBO(BaseVBO):
    def __init__(self, ctx, vertex_array):
        self.vertex_array = vertex_array
        super().__init__(ctx)
        self.format = '3f 1f 3f 3f'
        self.attribs = ['in_color','in_opacity', 'in_normal', 'in_position']
    
    def update(self,vertex_array):
        self.destroy()
        self.vertex_array =vertex_array
        self.vbo = self.get_vbo()


















