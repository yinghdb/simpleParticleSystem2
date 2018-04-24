#define FREEGLUT_STATIC
#define GLEW_STATIC

#include "simple_particle.cuh"
#include <curand.h>
#include <stdio.h>      
#include <stdlib.h>     
#include "GL\glew.h"
#include "GL\glut.h"
#include <time.h>
#include <cuda_gl_interop.h>



GLuint bufferObj;
cudaGraphicsResource *resource;
int term = 0;
uchar4* devPtr;
clock_t CPU_time;
static simpleParticleSystem sps;

void init_particle_system_point(simpleParticleSystem &sps)
{
	sps.TYPE = PointGenerator;
	sps.generator_center = make_float2(DIM_X / 2, DIM_Y / 2);

	sps.MAX_PARTICLE_SIZE = 12800;
	sps.ONE_BATCH_PARTICLE_SIZE = 64;
	sps.MAX_VELOCITY = 20.0;
	sps.MIN_VELOCITY = 5.0;
	sps.LIFE_TIME = 2.2;
	sps.INIT_RADIUS = 4;
	sps.VORTEX_WIDTH = 50;
	sps.VORTEX_HEIGHT = 50;
	sps.MAP_WIDTH = DIM_X;
	sps.MAP_HEIGHT = DIM_Y;
}

void randomGenerator(float* devData, int number, unsigned long long seed) {
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
	curandGenerateUniform(gen, devData, number);
	curandDestroyGenerator(gen);
}

void particle_canvas_update() {
	// 映射该共享资源 
	cudaGraphicsMapResources(1, &resource, NULL);
	// 请求一个指向映射资源的指针
	size_t size;
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

	render_particles(devPtr, DIM_X, DIM_Y);

	// 取消映射，确保cudaGraphicsUnmapResource()之前的所有CUDA操作完成
	cudaGraphicsUnmapResources(1, &resource, NULL);
}

//openGL render functions
void drawFunc(void)
{
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(DIM_X, DIM_Y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

static void keyFunc(unsigned char key, int x, int y)
{
	switch (key) {
	case 27:
		cudaGraphicsUnregisterResource(resource);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bufferObj);
		exit(0);
	}
}

void idleFunc(void) {
	clock_t CPU_time_old = CPU_time;
	CPU_time = clock();
	double fps = 1000.0 / (CPU_time - CPU_time_old);
	float passed_time = (CPU_time - CPU_time_old) / 1000.0;
	printf("fps: %f\n", fps);

	term += 1;

	//init batch particles
	randomGenerator(sps.rand_data, 3 * sps.ONE_BATCH_PARTICLE_SIZE, (unsigned long long)CPU_time*1000);
	generate_particles(sps.ONE_BATCH_PARTICLE_SIZE);

	//update old particles
	updata_particles(sps.ONE_BATCH_PARTICLE_SIZE, passed_time);

	//update canvas
	particle_canvas_update();

	glutPostRedisplay();
}

int init_cuda_gl(int argc, char* argv[]) {
	// 定义一个设备属性对象prop    
	cudaDeviceProp prop;
	int dev;

	memset(&prop, 0, sizeof(cudaDeviceProp));

	//限定设备计算功能集的版本号    
	prop.major = 1;
	prop.minor = 0;

	//选择在计算功能集的版本号为1.0的GPU设备上运行    
	cudaChooseDevice(&dev, &prop);

	//选定GL程序运行的设备    
	cudaGLSetGLDevice(dev);

	//OpenGL环境初始化    
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(DIM_X, DIM_Y);
	glutCreateWindow("CUDA+OpenGL");

	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		return -1;
	}

	// 创建像素缓冲区对象
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM_X*DIM_Y * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	// imgId运行时将在CUDA和OpenGL间共享，通过把imgId注册为一个图形资源
	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
}



int main(int argc, char* argv[]) {

	//init meta data
	init_particle_system_point(sps);
	//gpu memory malloc
	init_particles_cuda(sps, DIM_X, DIM_Y);
	init_vortex_field(sps);
	//bound d_sps
	copy_to_device_sps(sps);

	//init opengl
	init_cuda_gl(argc, argv);

	//init batch particles
	randomGenerator(sps.rand_data, 3 * sps.ONE_BATCH_PARTICLE_SIZE, 12345LL);
	generate_particles(sps.ONE_BATCH_PARTICLE_SIZE, point_pos);

	//update canvas
	particle_canvas_update();

	glutKeyboardFunc(keyFunc);
	glutDisplayFunc(drawFunc);
	glutIdleFunc(idleFunc);
	glutMainLoop();

	//destory gpu memory
	destroy_particles_cuda(sps);


	return 0;
}