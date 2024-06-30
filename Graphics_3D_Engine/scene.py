from .model import *


class Scene:
    def __init__(self, app, object_data):
        self.app = app
        self.object_data = object_data
        self.objects = []
        self.u = 0
        self.v = 1
        self.load()

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object

        for obj in self.object_data:
            add(Triangles(app, pos=obj[1], scale=obj[2]))

    def render(self):
        i = 0
        for obj in self.objects:
            obj.render(self.object_data[i][0])
            i +=1
   