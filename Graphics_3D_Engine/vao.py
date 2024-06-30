from .vbo import VBO
from .shader_program import ShaderProgram


class VAO:
    def __init__(self, ctx, triangle_data):
        self.ctx = ctx
        self.vbo = VBO(ctx, triangle_data)
        self.program = ShaderProgram(ctx)
        self.vaos = {}

        # cube vao
        self.vaos['triangles'] = self.get_vao(
            program=self.program.programs['default'],
            vbo = self.vbo.vbos['triangles'])


    def get_vao(self, program, vbo):
        vao = self.ctx.vertex_array(program, [(vbo.vbo, vbo.format, *vbo.attribs)])
        return vao
    
    def update(self, triangle_data):
        self.vbo.vbos['triangles'].update(triangle_data)
        self.vaos['triangles'] = self.get_vao(
            program=self.program.programs['default'],
            vbo = self.vbo.vbos['triangles'])

    def destroy(self):
        self.vbo.destroy()
        self.program.destroy()