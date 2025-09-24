#version 330 core

layout(location=0)out vec4 fragColour0;
layout(location=1)out vec4 fragColour1;
layout(location=2)out vec4 fragColour2;
layout(location=3)out vec4 fragColour3;
layout(location=4)out vec4 fragColour4;
layout(location=5)out vec4 fragColour5;
layout(location=6)out vec4 fragColour6;
layout(location=7)out vec4 fragColour7;



uniform int width;
uniform int height;
uniform float checkSize;
float checker( vec2 uv )
{
  float v = floor( checkSize * uv.x ) +floor( checkSize * uv.y );
  if( mod( v, 2.0 ) < 1.0 )
     return 1.0;
  else
     return 0.0;

}


void main()
{
  vec2  uv=vec2(gl_FragCoord.x/width,gl_FragCoord.y/height);
  float io=checker(uv);
  fragColour0.rgb= io==1 ? vec3(1,0,0) : vec3(1,1,1);
  fragColour1.rgb= io==1 ? vec3(0,1,0) : vec3(1,1,1);
  fragColour2.rgb= io==1 ? vec3(0,0,1) : vec3(1,1,1);
  fragColour3.rgb= io==1 ? vec3(0,0,0) : vec3(1,1,1);
  fragColour4.rgb= io==1 ? vec3(1,0,1) : vec3(1,1,1);
  fragColour5.rgb= io==1 ? vec3(1,1,0) : vec3(0,0,0);
  fragColour6.rgb= io==1 ? vec3(0,0,1) : vec3(0,0,0);
  fragColour7.rgb= io==1 ? vec3(1,0,1) : vec3(0,0,0);



}
