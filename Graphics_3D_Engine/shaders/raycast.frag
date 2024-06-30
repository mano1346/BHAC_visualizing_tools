#version 330 core

layout (location = 0) out vec4 fragColor;

in vec3 uv_0;
in vec3 normal;
in vec3 fragPos;

struct Light {
    vec3 position;
    vec3 Ia;
    vec3 Id;
    vec3 Is;
};

uniform Light light;
uniform vec3 camPos;


vec3 getLight(vec3 color) {
    vec3 Normal = normalize(normal);

    // ambient light
    vec3 ambient = light.Ia;

    // diffuse light
    vec3 lightDir = normalize(light.position - fragPos);
    float diff = max(0, dot(lightDir, Normal));
    vec3 diffuse = diff * light.Id;

    // specular light
    vec3 viewDir = normalize(camPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, Normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0), 32);
    vec3 specular = spec * light.Is;

    return color * (ambient + diffuse + specular);
}

void render(inout vec3 col, in vec2 uv) {
    vec3 lookAt = vec3(0, 1, 0);
    vec3 rd = getCam(ro, lookAt) * normalize(vec3(uv, FOV));

    vec2 object = rayMarch(camPos, rd);

    if (object.x < MAX_DIST) {
        vec3 p = ro + object.x * rd;
        vec3 material = getMaterial(p, object.y);
        col += getLight(p, rd, material);
        // fog
        col = mix(col, 1.0 - exp(-0.00002 * object.x * object.x));
    } else {
        col += background - max(0.9 * rd.y, 0.0);
    }
}

for (int i = 0; i < 256; ++i) {  // Assuming max 256 steps
        vec4 sample = sample_volume(current_position);
        accumulated_color += sample * (1.0 - accumulated_color.a);  // Accumulate color
        current_position += ray_dir * step_size;
        if (accumulated_color.a >= 1.0) {
            break;  // Early termination if fully opaque
        }
    }
void main() {
    float gamma = 2.2;
    vec3 color = vec3(uv_0);
    color = pow(color, vec3(gamma));

    color = getLight(color);

    color = pow(color, 1 / vec3(gamma));
    fragColor = vec4(color, 1.0);
}










