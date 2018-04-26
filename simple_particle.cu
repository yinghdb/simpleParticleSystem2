//#include "simple_particle.cuh"
//#include "device_launch_parameters.h"
//#include "device_functions.h"
//#include "math_functions.h"
//#include "cuda_runtime.h"
//#include <stdio.h>
//
//__constant__ simpleParticleSystem d_sps[1];
//
//__global__ void generateParticles();
//
//__global__ void renderParticles(uchar4* devPtr, int img_width, int img_height);
//
//__global__ void renderParticles_2(uchar4* devPtr, int img_width, int img_height, int tile_height);
//
//__global__ void updateParticles(float passed_time);
//
//__device__ float2 get_normal_vector(float rand_num);
//
//__device__ float get_energy(float2 p1, float2 p2, float dist_bound_powerd);
//
//__device__ uchar4 get_color_from_energy(float energy);
//
//__device__ float2 get_acceleration(int index);
//
//__device__ void update_particle_velocity(int index, float2 acc, float passed_time);
//
//__device__ int update_particle_position(int index, float passed_time); //return whether the particle is dead
//
//void init_particles_cuda(simpleParticleSystem &sps) {
//	int max_num_particles = sps.MAX_PARTICLE_SIZE;
//	int one_batch_num_particles = sps.ONE_BATCH_PARTICLE_SIZE;
//
//	cudaMalloc((void**)&sps.energy, sizeof(*sps.energy)*max_num_particles);
//	cudaMalloc((void**)&sps.position, sizeof(*sps.position)*max_num_particles);
//	cudaMalloc((void**)&sps.velocity, sizeof(*sps.velocity)*max_num_particles);
//	cudaMalloc((void**)&sps.remain_time, sizeof(*sps.remain_time)*max_num_particles);
//	cudaMalloc((void**)&sps.rand_data, sizeof(*sps.rand_data)*one_batch_num_particles*3);
//
//	cudaError_t err = cudaGetLastError();
//	if (err != cudaSuccess)
//		printf("Memory Allocation Error: %s\n", cudaGetErrorString(err));
//}
//
//void destroy_particles_cuda(simpleParticleSystem &sps) {
//	cudaError_t er;
//
//	er = cudaFree(sps.energy);
//	er = cudaFree(sps.position);
//	er = cudaFree(sps.velocity);
//	er = cudaFree(sps.remain_time);
//	er = cudaFree(sps.rand_data);
//
//	cudaError_t err = cudaGetLastError();
//	if (err != cudaSuccess)
//		printf("Memory Free Error: %s\n", cudaGetErrorString(err));
//}
//
//void copy_to_device_sps(simpleParticleSystem &sps) {
//	cudaError_t err = cudaMemcpyToSymbol(d_sps, &sps, sizeof(simpleParticleSystem));
//
//	if (err != cudaSuccess)
//		printf("Constant Memory Copy Error: %s\n", cudaGetErrorString(err));
//}
//
//void generate_particles(int generate_size) {
//	generateParticles << < 1, generate_size >> > ();
//	//generateParticlesLine <<< 1, sps.ONE_BATCH_PARTICLE_SIZE >>> (
//	//	sps.position, sps.velocity_orientation, sps.velocity, sps.remain_time, sps.rand_data, sps.ONE_BATCH_PARTICLE_SIZE,
//	//	sps.MAX_PARTICLE_SIZE, sps.generator_line[0], sps.generator_line[1], sps.MAX_VELOCITY, sps.MIN_VELOCITY, sps.LIFE_TIME
//	//);
//}
//
//void updata_particles(int generate_size, float passed_time) {
//	updateParticles << < 1, generate_size >> > (passed_time);
//}
//
//void render_particles(uchar4* devPtr, int img_width, int img_height, int generate_size, int tile_height) {
//	int grid_size = (img_height + tile_height - 1) / tile_height;
//	renderParticles_2 << <grid_size, generate_size, img_width*tile_height*sizeof(float) >> > (devPtr, img_width, img_height, tile_height);
//
//	//int thread_dim = 16;
//	//int grid_dim_x = (img_width + thread_dim - 1) / thread_dim;
//	//int grid_dim_y = (img_height + thread_dim - 1) / thread_dim;
//	//dim3 grids(grid_dim_x, grid_dim_y);
//	//dim3 threads(thread_dim, thread_dim);
//	//renderParticles << <grids, threads >> > (devPtr, img_width, img_height);
//}
//
//__global__ void generateParticles()
//{
//	float2 *position = (*d_sps).position;
//	float2 *velocity = (*d_sps).velocity;
//	float *remain_time = (*d_sps).remain_time;
//	float *rand = (*d_sps).rand_data;
//	int generate_size = (*d_sps).ONE_BATCH_PARTICLE_SIZE;
//	int max_size = (*d_sps).MAX_PARTICLE_SIZE;
//
//	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
//
//	__shared__ unsigned int generate_start_index;
//
//	//get the particle generate block pos
//	if (index == 0) {
//		max_size -= generate_size;
//		generate_start_index = 0;
//		while (generate_start_index <= max_size) {
//			if (remain_time[generate_start_index] == 0)
//				break;
//			generate_start_index += generate_size;
//		}
//	}
//
//	__syncthreads();
//
//	if (generate_start_index > max_size)
//		return;
//
//	int pid = generate_start_index + index; 
//	float x;
//	float y;
//	float2 velocity_orientation;
//	float n_velocity;
//
//	//generate rand position and velocity
//	switch ((*d_sps).TYPE)
//	{
//	case LineGenerator:
//		x = rand[index] * ((*d_sps).generator_line[0].x - (*d_sps).generator_line[1].x) + (*d_sps).generator_line[1].x;
//		y = rand[index] * ((*d_sps).generator_line[0].y - (*d_sps).generator_line[1].y) + (*d_sps).generator_line[1].y;
//		position[pid] = make_float2(x, y);
//
//		rand += generate_size;
//		pid = generate_start_index + index;
//		velocity_orientation = get_normal_vector(rand[index]);
//
//		rand += generate_size;
//		n_velocity = rand[index] * ((*d_sps).MAX_VELOCITY - (*d_sps).MIN_VELOCITY) + (*d_sps).MIN_VELOCITY;
//		velocity[pid].x = n_velocity * velocity_orientation.x;
//		velocity[pid].y = n_velocity * velocity_orientation.y;
//		break;
//	case CircleGenerator:
//		float rand_pos = rand[index];
//		float2 vec = get_normal_vector(rand_pos);
//		x = vec.x * (*d_sps).generator_radius.x + (*d_sps).generator_center.x;
//		y = vec.y * (*d_sps).generator_radius.y + (*d_sps).generator_center.y;
//		position[pid] = make_float2(x, y);
//
//		rand += generate_size;
//		pid = generate_start_index + index;
//		float rand_orient = rand[index];
//		rand_orient = rand_pos + (rand_orient / 2 - rand_orient / 4);
//		velocity_orientation = get_normal_vector(rand_orient);
//
//		rand += generate_size;
//		n_velocity = rand[index] * ((*d_sps).MAX_VELOCITY - (*d_sps).MIN_VELOCITY) + (*d_sps).MIN_VELOCITY;
//		velocity[pid].x = n_velocity * velocity_orientation.x;
//		velocity[pid].y = n_velocity * velocity_orientation.y;
//		break;
//	default:
//		break;
//	}
//
//	//generate remain time
//	remain_time[pid] = (*d_sps).LIFE_TIME;
//}
//
//__global__ void renderParticles_2(uchar4* devPtr, int img_width, int img_height, int tile_height) {
//	unsigned int index = threadIdx.x;
//	unsigned int strip = blockDim.x;
//	unsigned int start_index = 0;
//
//	extern __shared__ float energy_map[];
//
//	unsigned int start_height = blockIdx.x * tile_height;
//	unsigned int end_height = (blockIdx.x + 1) * tile_height;
//	if (end_height > img_height)
//		end_height = img_height;
//
//	int offset = threadIdx.x;
//	while (offset < (end_height - start_height)*img_width) {
//		energy_map[offset] = 0;
//		offset += blockDim.x;
//	}
//
//	__syncthreads();
//
//	float energy_scope = (*d_sps).ENERGY_SCOPE;
//	float dist_bound_powerd = energy_scope * energy_scope;
//	while (index < (*d_sps).MAX_PARTICLE_SIZE) {
//		if ((*d_sps).remain_time[start_index] != 0) {
//			if (index != start_index) {
//				float2 pos = (*d_sps).position[index];
//				float start_y = pos.y - energy_scope;
//				float end_y = pos.y + energy_scope;
//				if (start_y < start_height)
//					start_y = start_height;
//				if (end_y > end_height)
//					end_y = end_height;
//				for (float y = start_y; y < end_y; y += 1) {
//					for (float dx = -energy_scope; dx <= energy_scope; dx += 1) {
//						float2 near_pos = make_float2(pos.x + dx, y);
//						float energy = get_energy(pos, near_pos, dist_bound_powerd);
//						if (energy != 0) {
//							int px = int(near_pos.x);
//							int py = int(near_pos.y);
//							if (px < 0)
//								px = 0;
//							if (px >= img_width)
//								px = img_width - 1;
//							atomicAdd(&energy_map[(py-start_height)*img_width + px], energy);
//						}
//					}
//				}
//			}
//		}
//
//		index += strip;
//		start_index += strip;
//	}
//
//	__syncthreads();
//
//	offset = threadIdx.x;
//	int offset_dev = threadIdx.x + start_height * img_width;
//	while (offset < (end_height - start_height)*img_width) {
//		float energy = energy_map[offset];
//		if (energy > 1)
//			energy = 1;
//		devPtr[offset_dev] = get_color_from_energy(energy);
//
//		offset += blockDim.x;
//		offset_dev += blockDim.x;
//	}
//}
//
//__global__ void renderParticles(uchar4* devPtr, int img_width, int img_height) {
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//
//	if (x >= img_width || y >= img_height)
//		return;
//
//	
//	if (!(x >= (*d_sps).LIFE_BOUND[0] && x <= (*d_sps).LIFE_BOUND[2]
//		&& y <= (*d_sps).LIFE_BOUND[1] && y >= (*d_sps).LIFE_BOUND[3]))
//		return;
//
//	int generate_size = (*d_sps).ONE_BATCH_PARTICLE_SIZE;
//	int max_size = (*d_sps).MAX_PARTICLE_SIZE;
//	float energy = 0;
//	float dist_bound_powerd = (*d_sps).ENERGY_SCOPE * (*d_sps).ENERGY_SCOPE;
//	float2 pos = make_float2(x, y); 
//	for (int start_index = 0; start_index < max_size - generate_size; start_index += generate_size)
//	{
//		if ((*d_sps).remain_time[start_index] != 0) {
//			//here we do not render the first particle of the batch
//			for (int index = start_index + 1; index < start_index + generate_size; ++index) {
//				if ((*d_sps).remain_time[index] != 0) {
//					energy += get_energy((*d_sps).position[index], pos, dist_bound_powerd);
//					if (energy >= 1) {
//						energy = 1;
//						break;
//					}
//				}
//			}
//			if (energy >= 1) {
//				break;
//			}
//		}
//	}
//
//
//	int offset = x + y * img_width;
//	devPtr[offset] = get_color_from_energy(energy);
//}
//
//
//__global__ void updateParticles(float passed_time) {
//	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
//	unsigned int strip = gridDim.x * blockDim.x;
//	unsigned int start_index = 0;
//
//	__shared__ int living_particle_num;
//
//	while (index < (*d_sps).MAX_PARTICLE_SIZE) {
//		living_particle_num = 0;
//		//__syncthreads();
//
//		if ((*d_sps).remain_time[start_index] != 0) {
//			if (index != start_index) {
//				float2 acc = get_acceleration(index);
//				update_particle_velocity(index, acc, passed_time);
//				int is_living = update_particle_position(index, passed_time);
//				if (is_living) {
//					living_particle_num += 1;
//				}
//			}
//
//			__syncthreads();
//
//			if (index == start_index) {
//				if(living_particle_num == 0)
//					(*d_sps).remain_time[index] = 0;
//				else
//					(*d_sps).remain_time[index] = 1.0;
//			}
//		}
//
//		index += strip;
//		start_index += strip;
//	}
//}
//
//
//__device__ float2 get_normal_vector(float rand_num) {
//	float x, y;
//	sincosf(rand_num*2*PI, &y, &x);
//
//	return make_float2(x, y);
//}
//
//__device__ float get_energy(float2 p1, float2 p2, float dist_bound_powerd) {
//	float dx = p1.x - p2.x;
//	float dy = p1.y - p2.y;
//	float dist_powered = dx*dx + dy*dy;
//
//	if (dist_powered > dist_bound_powerd)
//		return 0;
//	if (dist_powered == 0)
//		return 0.05;
//	return 0.05 * sqrtf(dist_bound_powerd - dist_powered)/sqrtf(dist_bound_powerd);
//}
//
//__device__ uchar4 get_color_from_energy(float energy) {
//	if (energy == 0)
//		return make_uchar4(0, 0, 0, 0);
//
//	unsigned char r = 90 * energy + 160;
//	unsigned char g = 180 * energy;
//	unsigned char b = 60 * energy;
//	unsigned char w = 255 * energy;
//
//	return make_uchar4(r, g, b, w);
//}
//
//__device__ float2 get_acceleration(int index) {
//	float2 pos = (*d_sps).position[index];
//	float acc_x = (((*d_sps).LIFE_BOUND[0] + (*d_sps).LIFE_BOUND[2]) / 2 - pos.x) * 1.0;
//
//	return make_float2(acc_x, 80.0);
//}
//
//__device__ void update_particle_velocity(int index, float2 acc, float passed_time) {
//	(*d_sps).velocity[index].x += acc.x * passed_time;
//	(*d_sps).velocity[index].y += acc.y * passed_time;
//}
//
//__device__ int update_particle_position(int index, float passed_time) {
//	(*d_sps).remain_time[index] -= passed_time;
//	if ((*d_sps).remain_time[index] <= 0) {
//		(*d_sps).remain_time[index] = 0;
//		return 0;
//	}
//	
//	float2 *pos = &(*d_sps).position[index];
//	(*pos).x += (*d_sps).velocity[index].x * passed_time;
//	(*pos).y += (*d_sps).velocity[index].y * passed_time;
//	float x = (*pos).x;
//	float y = (*pos).y;
//
//	if (x > (*d_sps).LIFE_BOUND[0] && x < (*d_sps).LIFE_BOUND[2]
//		&& y < (*d_sps).LIFE_BOUND[1] && y > (*d_sps).LIFE_BOUND[3]) {
//
//		//if (x < (*d_sps).BOUND_BOX[0] + (*d_sps).ENERGY_SCOPE)
//		//	(*d_sps).BOUND_BOX[0] = x - (*d_sps).ENERGY_SCOPE;
//		//else if (x >(*d_sps).BOUND_BOX[2] - (*d_sps).ENERGY_SCOPE)
//		//	(*d_sps).BOUND_BOX[2] = x + (*d_sps).ENERGY_SCOPE;
//		//if (y < (*d_sps).BOUND_BOX[3] + (*d_sps).ENERGY_SCOPE)
//		//	(*d_sps).BOUND_BOX[3] = y - (*d_sps).ENERGY_SCOPE;
//		//else if (y >(*d_sps).BOUND_BOX[1] - (*d_sps).ENERGY_SCOPE)
//		//	(*d_sps).BOUND_BOX[1] = y + (*d_sps).ENERGY_SCOPE;
//
//		return 1;
//	}
//
//	(*d_sps).remain_time[index] = 0;
//	return 0;
//}

#include "simple_particle.cuh"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "math_functions.h"
#include <stdio.h>

__constant__ simpleParticleSystem d_sps[1];

__global__ void generateParticles(float2 point_pos);

__global__ void renderParticles(uchar4* devPtr, int img_width, int img_height);

__global__ void updateParticles(float passed_time);

__device__ void draw_circle(uchar4* devPtr, float *map_remain, float2 pos, float size, uchar4 color, float remain_time, int img_width, int img_height);

__device__ float2 get_normal_vector(float rand_num);

__device__ float2 get_acceleration(float2 position, simpleParticleSystem *sps, int pid);

__device__ void update_particle_velocity(float2 &velocity, float2 acc, float passed_time);

__device__ int update_particle_position(int index, float passed_time); //return whether the particle is dead

__device__ void update_particle_size(simpleParticleSystem *sps, float resize_speed, int pid, float passed_time);

void init_particles_cuda(simpleParticleSystem &sps, int image_width, int image_height) {
	int max_num_particles = sps.MAX_PARTICLE_NUM;
	int one_batch_num_particles = sps.ONE_BATCH_PARTICLE_NUM;

	cudaMalloc((void**)&sps.position, sizeof(*sps.position)*max_num_particles);
	cudaMalloc((void**)&sps.velocity, sizeof(*sps.velocity)*max_num_particles);
	cudaMalloc((void**)&sps.radius, sizeof(*sps.radius)*max_num_particles);
	cudaMalloc((void**)&sps.color_rgba, sizeof(*sps.color_rgba)*max_num_particles);
	cudaMalloc((void**)&sps.remain_time, sizeof(*sps.remain_time)*max_num_particles);
	cudaMalloc((void**)&sps.rand_data, sizeof(*sps.rand_data)*one_batch_num_particles * 4);
	cudaMalloc((void**)&sps.map_remain, sizeof(*sps.map_remain)*sps.MAP_WIDTH*sps.MAP_HEIGHT);
	cudaMalloc((void**)&sps.vortex_field, sizeof(*sps.vortex_field)*sps.VORTEX_WIDTH*sps.VORTEX_HEIGHT);
	cudaMalloc((void**)&sps.influence_factor, sizeof(*sps.influence_factor)*max_num_particles);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Memory Allocation Error: %s\n", cudaGetErrorString(err));
}

void init_vortex_field(simpleParticleSystem &sps) {
	float2 *h_vortex = new float2[sps.VORTEX_WIDTH*sps.VORTEX_HEIGHT];

	int index = 0;
	for (int y = 0; y < sps.VORTEX_HEIGHT; ++y) {
		for (int x = 0; x < sps.VORTEX_WIDTH; ++x) {
			float2 v0 = make_float2(x - sps.VORTEX_WIDTH / 2, y - sps.VORTEX_HEIGHT / 2);
			float2 v1 = make_float2(v0.y, -v0.x);
			h_vortex[index] = make_float2((v1.x - v0.x / 2) / (sps.VORTEX_WIDTH / 2), (v1.y - v0.y / 2) / (sps.VORTEX_HEIGHT / 2));
			index += 1;
		}
	}
	cudaMemcpy(sps.vortex_field, h_vortex, sps.VORTEX_WIDTH*sps.VORTEX_HEIGHT * sizeof(*h_vortex), cudaMemcpyHostToDevice);
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
	er = cudaFree(sps.influence_factor);

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
	cudaMemset((*d_sps).map_remain, 0, img_width*img_height * sizeof(*((*d_sps).map_remain)));
	cudaMemset(devPtr, 0, img_width*img_height * sizeof(*devPtr));

	int thread_dim = 24;
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
	float *influence_factor = (*d_sps).influence_factor;
	int generate_size = (*d_sps).ONE_BATCH_PARTICLE_NUM;
	int max_size = (*d_sps).MAX_PARTICLE_NUM;

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

	if (generate_start_index > max_size) {
		printf("generate manle\n");
		return;
	}

	int pid = generate_start_index + index;
	float x;
	float y;
	float2 velocity_orientation;
	float n_velocity;

	// position
	position[pid] = point_pos;
	//generate rand color and velocity
	float rand_color = rand[index];
	if (rand_color > 0.5) {
		color_rgba[pid] = make_uchar4(255, 255, 0, 128);
	}
	else if (rand_color <= 0.5) {
		color_rgba[pid] = make_uchar4(0, 255, 255, 128);
	}

	rand += generate_size;
	float rand_orient = rand[index];
	velocity_orientation = get_normal_vector(rand_orient);

	rand += generate_size;
	n_velocity = rand[index] * ((*d_sps).MAX_VELOCITY - (*d_sps).MIN_VELOCITY) + (*d_sps).MIN_VELOCITY;
	velocity[pid].x = n_velocity * velocity_orientation.x;
	velocity[pid].y = n_velocity * velocity_orientation.y;

	//generate rand influence_factor, size and remain time
	rand += generate_size;
	//printf("rand: %f", rand[index]);

	influence_factor[pid] = rand[index] * ((*d_sps).MAX_INFLU_FACTOR - (*d_sps).MIN_INFLU_FACTOR) + (*d_sps).MIN_INFLU_FACTOR;
	radius[pid] = rand[index] * ((*d_sps).MAX_RADIUS - (*d_sps).MIN_RADIUS) + (*d_sps).MIN_RADIUS;
	remain_time[pid] = rand[index] * ((*d_sps).MAX_LIFE_TIME - (*d_sps).MIN_LIFE_TIME) + (*d_sps).MIN_LIFE_TIME;


}

__global__ void renderParticles(uchar4* devPtr, int img_width, int img_height) {
	int batch_size = (*d_sps).ONE_BATCH_PARTICLE_NUM;
	int max_size = (*d_sps).MAX_PARTICLE_NUM;

	int batch_inner_start_id = threadIdx.x;
	int batch_inner_step = blockDim.x;
	int batch_outer_start_id = blockIdx.x * batch_size;
	int batch_outer_step = gridDim.x * batch_size;

	for (int batch_start_id = batch_outer_start_id; batch_start_id < max_size - batch_outer_step; batch_start_id += batch_outer_step) {
		if ((*d_sps).remain_time[batch_start_id] > 0) {
			for (int inner_id = batch_inner_start_id; inner_id < batch_size; inner_id += batch_inner_step) {
				if (inner_id == 0)
					continue;

				int id = batch_start_id + inner_id;
				float this_remain_time = (*d_sps).remain_time[id];
				if (this_remain_time > 0) {
					float* map_remain = (*d_sps).map_remain;
					float2 pos = (*d_sps).position[id];
					float size = (*d_sps).radius[id];
					uchar4 color = (*d_sps).color_rgba[id];
					float remain_time = (*d_sps).remain_time[id];

					//printf("position: %f, %f size: %f, color: %d %d %d %d \n", pos.x, pos.y, size, color.x, color.y, color.z, color.w);

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

	while (index < (*d_sps).MAX_PARTICLE_NUM) {
		living_particle_num = 0;
		//__syncthreads();

		// printf("index: %d, remain time: %f\n", index, (*d_sps).remain_time[index]);
		if ((*d_sps).remain_time[start_index] != 0) {
			if (index != start_index) {
				update_particle_size(d_sps, -0.4, index, passed_time);
				float2 acc = get_acceleration((*d_sps).position[index], d_sps, index);
				//printf("acc %f %f\n", acc.x, acc.y);
				update_particle_velocity((*d_sps).velocity[index], acc, passed_time);
				int is_living = update_particle_position(index, passed_time);
				if (is_living) {
					living_particle_num += 1;
				}
			}

			__syncthreads();

			if (index == start_index) {
				if (living_particle_num == 0)
					(*d_sps).remain_time[index] = 0;
				else
					(*d_sps).remain_time[index] = 1.0;
			}
		}

		index += strip;
		start_index += strip;
	}
}

__device__ void draw_circle(uchar4* devPtr, float *map_remain, float2 pos, float size, uchar4 color, float remain_time, int img_width, int img_height) {
	float size_square = size * size;
	for (int y = pos.y - size; y < pos.y + size + 1; ++y) {
		if (y < 0 || y >= img_height)
			continue;
		for (int x = pos.x - size; x < pos.x + size + 1; ++x) {
			if (x < 0 || x >= img_width)
				continue;
			float dist_square = (x - pos.x)*(x - pos.x) + (y - pos.y)*(y - pos.y);
			if (dist_square < size_square) {
				int offset = x + y * img_width;
				//if (map_remain[offset] < remain_time) {
					map_remain[offset] = remain_time;
					devPtr[offset] = color;
				//}
			}
		}
	}
}


__device__ float2 get_normal_vector(float rand_num) {
	float x, y;
	sincosf(rand_num * 2 * PI, &y, &x);

	return make_float2(x, y);
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

__device__ float2 get_acceleration(float2 position, simpleParticleSystem *sps, int pid) {
	int vortex_x = position.x / sps->MAP_WIDTH * (sps->VORTEX_WIDTH);
	int vortex_y = position.y / sps->MAP_HEIGHT * (sps->VORTEX_HEIGHT);
	int index = vortex_y * sps->VORTEX_WIDTH + vortex_x;
	float2 acc = make_float2(sps->vortex_field[index].x * sps->influence_factor[pid], sps->vortex_field[index].y * sps->influence_factor[pid]);
	
	//printf("address ^%d: acc0 %f %f\n", sps, sps->vortex_field[index].x, sps->influence_factor[index]);
	return acc;
}

__device__ void update_particle_velocity(float2 &velocity, float2 acc, float passed_time) {
	velocity.x += acc.x * passed_time;
	velocity.y += acc.y * passed_time;
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

	//printf("velocity: %f, %f remain time: %f \n", (*d_sps).velocity[index].x, (*d_sps).velocity[index].y, (*d_sps).remain_time[index]);

	if (x < (*d_sps).MAP_WIDTH && x >= 0
		&& y >= 0 && y < (*d_sps).MAP_HEIGHT) {
		return 1;
	}

	(*d_sps).remain_time[index] = 0;
	return 0;
}

__device__ void update_particle_size(simpleParticleSystem *sps, float resize_speed, int pid, float passed_time) {
	sps->radius[pid] += resize_speed * passed_time;
	if (sps->radius[pid] < 1) {
		sps->radius[pid] = 1;
	}
}
