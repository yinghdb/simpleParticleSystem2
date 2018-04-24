#include "simple_particle.cuh"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "math_functions.h"
#include <stdio.h>

__constant__ simpleParticleSystem d_sps[1];

__global__ void generateParticles(float point_pos);

__global__ void renderParticles(uchar4* devPtr, int img_width, int img_height);

__global__ void updateParticles(float passed_time);

__device__ void draw_circle(uchar4* devPtr, float *map_remain, float2 pos, float size, uchar4 color, float remain_time, int img_width, int img_height);

__device__ float2 get_normal_vector(float rand_num);

__device__ uchar4 get_color_from_energy(float energy);

__device__ float2 get_acceleration(float2 position, simpleParticleSystem *sps);

__device__ void update_particle_velocity(float2 &velocity, float2 acc, float passed_time);

__device__ int update_particle_position(int index, float passed_time); //return whether the particle is dead

void init_particles_cuda(simpleParticleSystem &sps, int image_width, int image_height) {
	int max_num_particles = sps.MAX_PARTICLE_SIZE;
	int one_batch_num_particles = sps.ONE_BATCH_PARTICLE_SIZE;

	cudaMalloc((void**)&sps.position, sizeof(*sps.position)*max_num_particles);
	cudaMalloc((void**)&sps.velocity, sizeof(*sps.velocity)*max_num_particles);
	cudaMalloc((void**)&sps.radius, sizeof(*sps.radius)*max_num_particles);
	cudaMalloc((void**)&sps.color_rgba, sizeof(*sps.color_rgba)*max_num_particles);
	cudaMalloc((void**)&sps.remain_time, sizeof(*sps.remain_time)*max_num_particles);
	cudaMalloc((void**)&sps.rand_data, sizeof(*sps.rand_data)*one_batch_num_particles*3);
	cudaMalloc((void**)&sps.map_remain, sizeof(*sps.map_remain)*image_width*image_height);
	cudaMalloc((void**)&sps.vortex_field, sizeof(*sps.vortex_field)*sps.VORTEX_WIDTH*sps.VORTEX_HEIGHT);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Memory Allocation Error: %s\n", cudaGetErrorString(err));
}

void init_vortex_field(simpleParticleSystem &sps){
	float2 *h_vortex = new float2[sps.VORTEX_WIDTH*sps.VORTEX_HEIGHT];

	int index = 0;
	for (int y = 0; y < sps.VORTEX_HEIGHT; ++y){
		for(int x = 0; x < sps.VORTEX_WIDTH; ++x){
			float2 v0 = make_float2(x-sps.VORTEX_WIDTH/2, y-sps.VORTEX_HEIGHT/2);
			float2 v1 = make_float2(v0.y, -v0.x);
			h_vortex[index] = (v1 - v0/2) / (sps.VORTEX_WIDTH/2);
			index += 1;
		}
	}
	cudaMemcpy(sps.vortex_field, h_vortex, sps.VORTEX_WIDTH*sps.VORTEX_HEIGHT*sizeof(float2), cudaMemcpyHostToDevice);
	delete[] h_vortex;
}

void destroy_particles_cuda(simpleParticleSystem &sps) {
	cudaError_t er;

	er = cudaFree(sps.position);
	er = cudaFree(sps.velocity);
	er = cudaFree(sps.radius);
	er = cudaFree(sps.color_rgba);
	er = cudaFree(sps.remain_time);
	er = cudaFree(sps.rand_data);
	er = cudaFree(sps.map_remain);
	er = cudaFree(sps.vortex_field);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Memory Free Error: %s\n", cudaGetErrorString(err));
}

void copy_to_device_sps(simpleParticleSystem &sps) {
	cudaError_t err = cudaMemcpyToSymbol(d_sps, &sps, sizeof(simpleParticleSystem));

	if (err != cudaSuccess)
		printf("Constant Memory Copy Error: %s\n", cudaGetErrorString(err));
}

void generate_particles(int generate_size, float2 point_pos) {
	generateParticles << < 1, generate_size >> > (point_pos);
	//generateParticlesLine <<< 1, sps.ONE_BATCH_PARTICLE_SIZE >>> (
	//	sps.position, sps.velocity_orientation, sps.velocity, sps.remain_time, sps.rand_data, sps.ONE_BATCH_PARTICLE_SIZE,
	//	sps.MAX_PARTICLE_SIZE, sps.generator_line[0], sps.generator_line[1], sps.MAX_VELOCITY, sps.MIN_VELOCITY, sps.LIFE_TIME
	//);
}

void updata_particles(int generate_size, float passed_time) {
	updateParticles << < 1, generate_size >> > (passed_time);
}

void render_particles(uchar4* devPtr, int img_width, int img_height) {
	cudaMemset(sps.map_remain, 0, image_width*image_height);

	int thread_dim = 16;
	int grid_dim = 16;
	dim3 grids(grid_dim);
	dim3 threads(thread_dim);
	renderParticles << <grids, threads >> > (devPtr, img_width, img_height);
}

__global__ void generateParticles(float2 point_pos)
{
	float2 *position = (*d_sps).position;
	float2 *velocity = (*d_sps).velocity;
	float *radius = (*d_sps).radius;
	uchar4 *color_rgba = (*d_sps).color_rgba;
	float *remain_time = (*d_sps).remain_time;
	float *rand = (*d_sps).rand_data;
	int generate_size = (*d_sps).ONE_BATCH_PARTICLE_SIZE;
	int max_size = (*d_sps).MAX_PARTICLE_SIZE;

	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ unsigned int generate_start_index;

	//get the particle generate block pos
	if (index == 0) {
		max_size -= generate_size;
		generate_start_index = 0;
		while (generate_start_index <= max_size) {
			if (remain_time[generate_start_index] == 0)
				break;
			generate_start_index += generate_size;
		}
	}

	__syncthreads();

	if (generate_start_index > max_size)
		return;

	int pid = generate_start_index + index; 
	float x;
	float y;
	float2 velocity_orientation;
	float n_velocity;

	// position
	position[pid] = point_pos;
	radius[pid] = (*sps).INIT_RADIUS;
	//generate rand color and velocity
	switch ((*d_sps).TYPE)
	{
		float rand_color = rand[index];
		if(rand_color > 0.5){
			color_rgba[pid] = make_uchar4(1.0,1.0,0.0,1.0);
		}
		else if(rand_color <= 0.5){
			color_rgba[pid] = make_uchar4(0.0,1.0,1.0,1.0);
		}

		rand += generate_size;
		float rand_orient = rand[index];
		velocity_orientation = get_normal_vector(rand_orient);

		rand += generate_size;
		n_velocity = rand[index] * ((*d_sps).MAX_VELOCITY - (*d_sps).MIN_VELOCITY) + (*d_sps).MIN_VELOCITY;
		velocity[pid].x = n_velocity * velocity_orientation.x;
		velocity[pid].y = n_velocity * velocity_orientation.y;
		break;
	default:
		break;
	}

	//generate remain time
	remain_time[pid] = (*d_sps).LIFE_TIME;
}

__global__ void renderParticles(uchar4* devPtr, int img_width, int img_height) {
	int batch_size = (*d_sps).ONE_BATCH_PARTICLE_SIZE;
	int max_size = (*d_sps).MAX_PARTICLE_SIZE;

	int batch_inner_start_id = threadIdx.x;
	int batch_inner_step = blockDim.x;
	int batch_outer_start_id = blockIdx.x * batch_size;
	int batch_outer_step = blockDim.x * batch_size;



	for(int batch_start_id = batch_outer_start_id; batch_start_id < max_size - batch_outer_step; batch_start_id += batch_outer_step){
		if ((*d_sps).remain_time[start_index] > 0) {
			for(int inner_id = batch_inner_start_id; inner_id < batch_size; inner_id += batch_inner_step){
				if(inner_id == 0)
					continue;

				int id = batch_start_id + inner_id;
				float this_remain_time = (*d_sps).remain_time[index];
				if(this_remain_time > 0){
					float map_remain = (*d_sps).map_remain[index];
					float2 pos = (*d_sps).position[index];
					float size = (*d_sps).radius[index];
					uchar4 color = (*d_sps).color_rgba[index];
					draw_circle(devPtr, map_remain, pos, size, color, remain_time, img_width, img_height);
				}
			}
		}
	}
}

__global__ void updateParticles(float passed_time) {
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int strip = gridDim.x * blockDim.x;
	unsigned int start_index = blockIdx.x*blockDim.x;

	__shared__ int living_particle_num;

	while (index < (*d_sps).MAX_PARTICLE_SIZE) {
		living_particle_num = 0;
		//__syncthreads();

		if ((*d_sps).remain_time[start_index] != 0) {
			if (index != start_index) {
				float2 acc = get_acceleration((*d_sps).position[index], d_sps);
				update_particle_velocity(&((*d_sps).velocity[index]), acc, passed_time);
				int is_living = update_particle_position(index, passed_time);
				if (is_living) {
					living_particle_num += 1;
				}
			}

			__syncthreads();

			if (index == start_index) {
				if(living_particle_num == 0)
					(*d_sps).remain_time[index] = 0;
				else
					(*d_sps).remain_time[index] = 1.0;
			}
		}

		index += strip;
		start_index += strip;
	}
}

__device__ void draw_circle(uchar4* devPtr, float *map_remain, float2 pos, float size, uchar4 color, float remain_time, int img_width, int img_height){
	float size_square = size * size;
	for(int y = pos.y - size; y < pos.y + size + 1; ++y){
		if(y < 0 || y >= img_height)
			continue;
		for(int x = pos.x - size; x < pos.x + size + 1; ++x){
			if(x < 0 || x >= img_width)
				continue;
			float dist_square = (x-pos.x)*(x-pos.x) + (y-pos.y)*(y-pos.y);
			if(dist_square < size_square){
				int offset = x + y * img_width;
				if(map_remain[offset] < remain_time){
					map_remain[offset] = remain_time;
					devPtr[offset] = color;
				}
			}
		}
	}
}


__device__ float2 get_normal_vector(float rand_num) {
	float x, y;
	sincosf(rand_num*2*PI, &y, &x);

	return make_float2(x, y);
}

__device__ float get_energy(float2 p1, float2 p2, float dist_bound_powerd) {
	float dx = p1.x - p2.x;
	float dy = p1.y - p2.y;
	float dist_powered = dx*dx + dy*dy;

	if (dist_powered > dist_bound_powerd)
		return 0;
	if (dist_powered == 0)
		return 0.1;
	return 0.1 * sqrtf(dist_bound_powerd - dist_powered)/sqrtf(dist_bound_powerd);
}

__device__ uchar4 get_color_from_energy(float energy) {
	if (energy == 0)
		return make_uchar4(0, 0, 0, 0);

	unsigned char r = 90 * energy + 160;
	unsigned char g = 180 * energy;
	unsigned char b = 60 * energy;
	unsigned char w = 255 * energy;

	return make_uchar4(r, g, b, w);
}

__device__ float2 get_acceleration(float2 position, simpleParticleSystem *sps) {
	int vortex_x = position.x / sps->MAP_WIDTH * (sps->VORTEX_WIDTH);
	int vortex_y = position.y / sps->MAP_HEIGHT * (sps->VORTEX_HEIGHT);
	int index = vortex_y * img_width + vortex_x;
	float2 acc = sps->vortex_field[index] * 20;
	return acc;
}

__device__ void update_particle_velocity(float2 &velocity, float2 acc, float passed_time) {
	velocity->x += acc.x * passed_time;
	velocity->y += acc.y * passed_time;
}

__device__ int update_particle_position(int index, float passed_time) {
	(*d_sps).remain_time[index] -= passed_time;
	if ((*d_sps).remain_time[index] <= 0) {
		(*d_sps).remain_time[index] = 0;
		return 0;
	}
	
	float2 *pos = &(*d_sps).position[index];
	(*pos).x += (*d_sps).velocity[index].x * passed_time;
	(*pos).y += (*d_sps).velocity[index].y * passed_time;
	float x = (*pos).x;
	float y = (*pos).y;

	if (x > (*d_sps).LIFE_BOUND[0] && x < (*d_sps).LIFE_BOUND[2]
		&& y < (*d_sps).LIFE_BOUND[1] && y > (*d_sps).LIFE_BOUND[3]) {

		//if (x < (*d_sps).BOUND_BOX[0] + (*d_sps).ENERGY_SCOPE)
		//	(*d_sps).BOUND_BOX[0] = x - (*d_sps).ENERGY_SCOPE;
		//else if (x >(*d_sps).BOUND_BOX[2] - (*d_sps).ENERGY_SCOPE)
		//	(*d_sps).BOUND_BOX[2] = x + (*d_sps).ENERGY_SCOPE;
		//if (y < (*d_sps).BOUND_BOX[3] + (*d_sps).ENERGY_SCOPE)
		//	(*d_sps).BOUND_BOX[3] = y - (*d_sps).ENERGY_SCOPE;
		//else if (y >(*d_sps).BOUND_BOX[1] - (*d_sps).ENERGY_SCOPE)
		//	(*d_sps).BOUND_BOX[1] = y + (*d_sps).ENERGY_SCOPE;

		return 1;
	}

	(*d_sps).remain_time[index] = 0;
	return 0;
}