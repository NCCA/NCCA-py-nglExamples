#version 410 core
out vec4 Fragcolour;

in VS_OUT {
    vec3 fragPos;
    vec2 uv;
    vec3 tangentLightPos[3];
    vec3 tangentViewPos[3];
    vec3 tangentFragPos[3];
} fs_in;

uniform sampler2D diffuseMap;
uniform sampler2D spec;
uniform sampler2D normalMap;


struct Light
{
  vec3 position;
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};
uniform Light light[3];


void main()
{
  // lookup normal from normal map, move from [0,1] to [-1,1] range, normalize
  vec3 normal = texture(normalMap, fs_in.uv).xyz * 2.0 - 1.0;
  // flip Z if your normal maps use that convention (e.g. ZBrush)
  normal.z = -normal.z;
  normal = normalize(normal);

  // material/specular values (consider moving to uniforms or textures)
  vec3 specularMaterial = texture(spec, fs_in.uv).rgb;
  vec3 colour = texture(diffuseMap, fs_in.uv).rgb;

  vec3 ambient = vec3(0.0);
  vec3 diffuse = vec3(0.0);
  vec3 specular = vec3(0.0);

  for (int i = 0; i < 3; ++i)
  {
    // ambient contribution
    ambient += light[i].ambient * colour;

    // tangent-space light direction (tangentLightPos already = TBN * (lightPos - fragPos))
    vec3 lightDir = normalize(fs_in.tangentLightPos[i]);
    float LdotN = max(dot(lightDir, normal), 0.0);
    diffuse += LdotN * colour * light[i].diffuse;

    if (LdotN > 0.0)
    {
      // tangent-space view direction
      vec3 viewDir = normalize(fs_in.tangentViewPos[i]);
      // Blinn-Phong halfway vector
      vec3 halfwayDir = normalize(lightDir + viewDir);
      float spec = pow(max(dot(normal, halfwayDir), 0.0), 500.0);
      // use the corresponding light's specular term
      specular += specularMaterial * spec * light[i].specular;
    }
  }

  Fragcolour = vec4(ambient + diffuse + specular, 1.0);
  // Debug: visualize normal in tangent space
  // Fragcolour.rgb = normal * 0.5 + 0.5;
}
