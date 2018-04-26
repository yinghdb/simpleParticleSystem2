//#define FREEGLUT_STATIC
//#define GLEW_STATIC
//
//#include "simple_particle.cuh"
//#include <curand.h>
//#include <stdio.h>      
//#include <stdlib.h>     
//#include "GL\glew.h"
//#include "GL\glut.h"
//#include <time.h>
//#include <cuda_gl_interop.h>
//
//#define DIM_X 720
//#define DIM_Y 720
//#define MIN(x,y) (x>y?y:x)
//#define MAX(x,y) (x>y?x:y)
//
//GLuint bufferObj;
//cudaGraphicsResource *resource;
//int term = 0;
//uchar4* devPtr;
//clock_t CPU_time;
//static simpleParticleSystem sps;
//int cycle_count = 0;
//
//void init_particle_system_line(simpleParticleSystem &sps)
//{
//	sps.TYPE = LineGenerator;
//	sps.generator_line[0] = make_float2(DIM_X / 2, DIM_Y / 4);
//	sps.generator_line[1] = make_float2(DIM_X / 2, DIM_Y * 5 / 16);
//
//	sps.MAX_PARTICLE_SIZE = 3276800;
//	sps.ONE_BATCH_PARTICLE_SIZE = 256;
//	sps.ENERGY_SCOPE = 3.0;
//	sps.LIFE_BOUND[0] = DIM_X / 2 - 100;
//	sps.LIFE_BOUND[1] = DIM_Y / 2 * 50;
//	sps.LIFE_BOUND[2] = DIM_X / 2 + 100;
//	sps.LIFE_BOUND[3] = DIM_Y / 4 - 50;
//	sps.BOUND_BOX[0] = MIN(sps.generator_line[0].x, sps.generator_line[1].x) - sps.ENERGY_SCOPE;
//	sps.BOUND_BOX[1] = MAX(sps.generator_line[0].y, sps.generator_line[1].y) + sps.ENERGY_SCOPE;
//	sps.BOUND_BOX[2] = MAX(sps.generator_line[0].x, sps.generator_line[1].x) + sps.ENERGY_SCOPE;
//	sps.BOUND_BOX[3] = MIN(sps.generator_line[0].y, sps.generator_line[1].y) - sps.ENERGY_SCOPE;
//	sps.MAX_VELOCITY = 40.0;
//	sps.MIN_VELOCITY = 10.0;
//	sps.LIFE_TIME = 2.2;
//}
//
//void init_particle_system_circle(simpleParticleSystem &sps)
//{
//	sps.TYPE = CircleGenerator;
//	sps.generator_center = make_float2(360, 100);
//	sps.generator_radius = make_float2(40, 40);
//
//	sps.MAX_PARTICLE_SIZE = 1024000;
//	sps.ONE_BATCH_PARTICLE_SIZE = 1024;
//	sps.ENERGY_SCOPE = 2.0;
//	sps.LIFE_BOUND[0] = 0;
//	sps.LIFE_BOUND[1] = DIM_Y;
//	sps.LIFE_BOUND[2] = DIM_X;
//	sps.LIFE_BOUND[3] = 0;
//	sps.BOUND_BOX[0] = sps.generator_center.x - sps.generator_radius.x - sps.ENERGY_SCOPE;
//	sps.BOUND_BOX[1] = sps.generator_center.y + sps.generator_radius.y + sps.ENERGY_SCOPE;
//	sps.BOUND_BOX[2] = sps.generator_center.x + sps.generator_radius.x + sps.ENERGY_SCOPE;
//	sps.BOUND_BOX[3] = sps.generator_center.y - sps.generator_radius.y - sps.ENERGY_SCOPE;
//	sps.MAX_VELOCITY = 60.0;
//	sps.MIN_VELOCITY = 20.0;
//	sps.LIFE_TIME = 2.0;
//}
//
//void randomGenerator(float* devData, int number, unsigned long long seed) {
//	curandGenerator_t gen;
//	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
//	curandSetPseudoRandomGeneratorSeed(gen, seed);
//	curandGenerateUniform(gen, devData, number);
//	curandDestroyGenerator(gen);
//}
//
//void particle_canvas_update() {
//	// 映射该共享资源 
//	cudaGraphicsMapResources(1, &resource, NULL);
//	// 请求一个指向映射资源的指针
//	size_t size;
//	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);
//
//	render_particles(devPtr, DIM_X, DIM_Y, sps.ONE_BATCH_PARTICLE_SIZE, 8);
//
//	// 取消映射，确保cudaGraphicsUnmapResource()之前的所有CUDA操作完成
//	cudaGraphicsUnmapResources(1, &resource, NULL);
//}
//
////openGL render functions
//void drawFunc(void)
//{
//	glClearColor(0.0, 0.0, 0.0, 1.0);
//	glClear(GL_COLOR_BUFFER_BIT);
//	glDrawPixels(DIM_X, DIM_Y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
//	glutSwapBuffers();
//}
//
//static void keyFunc(unsigned char key, int x, int y)
//{
//	switch (key) {
//	case 27:
//		cudaGraphicsUnregisterResource(resource);
//		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
//		glDeleteBuffers(1, &bufferObj);
//		exit(0);
//	}
//}
//
//void idleFunc(void) {
//	clock_t CPU_time_old = CPU_time;
//	CPU_time = clock();
//	double fps = 1000.0 / (CPU_time - CPU_time_old);
//	float passed_time = (CPU_time - CPU_time_old) / 1000.0;
//	if (cycle_count % 25 == 0) {
//		printf("fps: %f\n", fps);
//	}
//	cycle_count++;
//
//	term += 1;
//
//	cudaEvent_t time0, time1, time2, time3;
//	float elapsedTime1, elapsedTime2, elapsedTime3;
//	cudaEventCreate(&time0);
//	cudaEventCreate(&time1);
//	cudaEventCreate(&time2);
//	cudaEventCreate(&time3);
//	cudaEventRecord(time0, 0);
//
//	//init batch particles
//	randomGenerator(sps.rand_data, 3 * sps.ONE_BATCH_PARTICLE_SIZE, (unsigned long long)CPU_time*1000);
//	generate_particles(sps.ONE_BATCH_PARTICLE_SIZE);
//
//	cudaEventRecord(time1, 0);
//	cudaEventSynchronize(time1);
//	cudaEventElapsedTime(&elapsedTime1, time0, time1);
//
//	//update old particles
//	updata_particles(sps.ONE_BATCH_PARTICLE_SIZE, passed_time);
//
//	cudaEventRecord(time2, 0);
//	cudaEventSynchronize(time2);
//	cudaEventElapsedTime(&elapsedTime2, time1, time2);
//
//	//update canvas
//	particle_canvas_update();
//
//	cudaEventRecord(time3, 0);
//	cudaEventSynchronize(time3);
//	cudaEventElapsedTime(&elapsedTime3, time2, time3);
//
//	if (cycle_count % 25 == 0) {
//		printf("generate time: %f; update time: %f; render time: %f; total: %f\n", (elapsedTime1), (elapsedTime2), (elapsedTime3), (elapsedTime1+elapsedTime2+elapsedTime3));
//	}
//
//	glutPostRedisplay();
//}
//
//int init_cuda_gl(int argc, char* argv[]) {
//	// 定义一个设备属性对象prop    
//	cudaDeviceProp prop;
//	int dev;
//
//	memset(&prop, 0, sizeof(cudaDeviceProp));
//
//	//限定设备计算功能集的版本号    
//	prop.major = 1;
//	prop.minor = 0;
//
//	//选择在计算功能集的版本号为1.0的GPU设备上运行    
//	cudaChooseDevice(&dev, &prop);
//
//	//选定GL程序运行的设备    
//	cudaGLSetGLDevice(dev);
//
//	//OpenGL环境初始化    
//	glutInit(&argc, argv);
//	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
//	glutInitWindowSize(DIM_X, DIM_Y);
//	glutCreateWindow("CUDA+OpenGL");
//
//	if (glewInit() != GLEW_OK) {
//		fprintf(stderr, "Failed to initialize GLEW\n");
//		getchar();
//		return -1;
//	}
//
//	// 创建像素缓冲区对象
//	glGenBuffers(1, &bufferObj);
//	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
//	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM_X*DIM_Y * 4, NULL, GL_DYNAMIC_DRAW_ARB);
//
//	// imgId运行时将在CUDA和OpenGL间共享，通过把imgId注册为一个图形资源
//	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
//}
//
//
//
//int main(int argc, char* argv[]) {
//
//	//init meta data
//	init_particle_system_line(sps);
//	//gpu memory malloc
//	init_particles_cuda(sps);
//	//bound d_sps
//	copy_to_device_sps(sps);
//
//	//init opengl
//	init_cuda_gl(argc, argv);
//
//	//init batch particles
//	randomGenerator(sps.rand_data, 3 * sps.ONE_BATCH_PARTICLE_SIZE, 12345LL);
//	generate_particles(sps.ONE_BATCH_PARTICLE_SIZE);
//
//	//update canvas
//	particle_canvas_update();
//
//	glutKeyboardFunc(keyFunc);
//	glutDisplayFunc(drawFunc);
//	glutIdleFunc(idleFunc);
//	glutMainLoop();
//
//	//destory gpu memory
//	destroy_particles_cuda(sps);
//
//
//	return 0;
//}

#define FREEGLUT_STATIC
#define GLEW_STATIC

#include "simple_particle.cuh"
#include <curand.h>
#include <stdio.h>      
#include <stdlib.h>     
#include "GL\glew.h"
#include "GL\glut.h"
#include <time.h>
#include <math.h>
#include <cuda_gl_interop.h>



GLuint bufferObj;
cudaGraphicsResource *resource;
int term = 0;
uchar4* devPtr;
clock_t CPU_time;
static simpleParticleSystem sps;

int click_state = GLUT_UP;
int last_x = 0;
int last_y = 0;

void init_particle_system_point(simpleParticleSystem &sps)
{
	sps.TYPE = PointGenerator;

	sps.MAX_PARTICLE_NUM = 1280000;
	sps.ONE_BATCH_PARTICLE_NUM = 24;
	sps.MAX_VELOCITY = 100.0;
	sps.MIN_VELOCITY = 0.0;
	sps.VORTEX_WIDTH = 50;
	sps.VORTEX_HEIGHT = 50;
	sps.MAP_WIDTH = DIM_X;
	sps.MAP_HEIGHT = DIM_Y;
	sps.MIN_INFLU_FACTOR = 1;
	sps.MAX_INFLU_FACTOR = 500;
	sps.MAX_LIFE_TIME = 10.0;
	sps.MIN_LIFE_TIME = 2.0;
	sps.MAX_RADIUS = 3.0;
	sps.MIN_RADIUS = 1.5;
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

static void mouseFunc(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		click_state = GLUT_DOWN;
		randomGenerator(sps.rand_data, 4 * sps.ONE_BATCH_PARTICLE_NUM, (unsigned long long)CPU_time * 1000);
		generate_particles(sps.ONE_BATCH_PARTICLE_NUM, make_float2(x, DIM_Y - y));
		last_x = x;
		last_y = y;
	}
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
		click_state = GLUT_UP;
	}
}

static void mouseMoveFunc(int x, int y) {
	if (click_state == GLUT_DOWN) {
		float dist = sqrt((last_x - x) * (last_x - x) + (last_y - y) * (last_y - y));
		if (dist > 20) {
			randomGenerator(sps.rand_data, 4 * sps.ONE_BATCH_PARTICLE_NUM, (unsigned long long)CPU_time * 1000);
			generate_particles(sps.ONE_BATCH_PARTICLE_NUM, make_float2(x, DIM_Y - y));
			last_x = x;
			last_y = y;
		}
	}
}

void idleFunc(void) {
	clock_t CPU_time_old = CPU_time;
	CPU_time = clock();
	double fps = 1000.0 / (CPU_time - CPU_time_old);
	float passed_time = (CPU_time - CPU_time_old) / 1000.0;
	printf("fps: %f\n", fps);

	term += 1;

	////init batch particles
	//randomGenerator(sps.rand_data, 4 * sps.ONE_BATCH_PARTICLE_NUM, (unsigned long long)CPU_time * 1000);
	//generate_particles(sps.ONE_BATCH_PARTICLE_NUM, make_float2(100, 100));

	//update old particles
	updata_particles(sps.ONE_BATCH_PARTICLE_NUM, passed_time);

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

	////init batch particles
	//randomGenerator(sps.rand_data, 4 * sps.ONE_BATCH_PARTICLE_NUM, 12345LL);
	//generate_particles(sps.ONE_BATCH_PARTICLE_NUM, make_float2(400, 400));

	////update canvas
	//particle_canvas_update();

	glutMotionFunc(mouseMoveFunc);
	glutMouseFunc(mouseFunc);
	glutKeyboardFunc(keyFunc);
	glutDisplayFunc(drawFunc);
	glutIdleFunc(idleFunc);
	glutMainLoop();

	//destory gpu memory
	destroy_particles_cuda(sps);
	

	return 0;
}