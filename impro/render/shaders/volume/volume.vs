#version 330 core

layout(location = 0) in vec3 vVertex; //object space vertex position

//uniform
uniform mat4 u_projection;
uniform mat4 u_modelview;   //combined modelview projection matrix

smooth out vec3 vUV; //3D texture coordinates for texture lookup in the fragment shader

void main()
{
	//get the clipspace position
	gl_Position = u_projection * u_modelview *vec4(vVertex.xyz,1);

	//get the 3D texture coordinates by adding (0.5,0.5,0.5) to the object space
	//vertex position. Since the unit cube is at origin (min: (-0.5,-0.5,-0.5) and max: (0.5,0.5,0.5))
	//adding (0.5,0.5,0.5) to the unit cube object space position gives us values from (0,0,0) to
	//(1,1,1)
	vUV = vVertex + vec3(0.5);
}