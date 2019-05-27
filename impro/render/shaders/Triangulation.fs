#version 330
in vec4 v_color;
//out vec4 Color;
layout(location = 0) out vec4 Color;



void main()
{
    Color = v_color;//vec4(pos)*v_color;
}