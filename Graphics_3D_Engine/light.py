import glm


class Light:
    def __init__(self, position=(0, 30, 10), color=(1, 1, 1)):
        self.position = glm.vec3(position)
        self.color = glm.vec3(color)
        # intensities
        self.Ia = 1.0 * self.color  # ambient
        self.Id = 0.0 * self.color  # diffuse
        self.Is = 0.0 * self.color  # specular