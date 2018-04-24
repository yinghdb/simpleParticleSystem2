#include "cuda_runtime.h"

#define PARTICLE_LIFE_MS 3000
#define PI 3.1415926
#define DIM_X 720
#define DIM_Y 720
#define MIN(x,y) (x>y?y:x)
#define MAX(x,y) (x>y?x:y)

enum ParticleGeneratorType
{
	LineGenerator,
	CircleGenerator,
	PointGenerator
};

typedef struct
{
	////physical attribute////
	float2 *position;
	float2 *velocity;
	float *radius;
	uchar4 *color_rgba;
	float *map_remain;
	float2 *vortex_field;

	////particle life////
	float *remain_time;

	////rand data used for generating particles////
	float *rand_data;

	////common attributes////
	int MAX_PARTICLE_NUM;
	int ONE_BATCH_PARTICLE_NUM;
	float MAX_VELOCITY; //pixels per second
	float MIN_VELOCITY;
	float LIFE_TIME; //second
	float INIT_RADIUS;
	int VORTEX_WIDTH;
	int VORTEX_HEIGHT;
	int MAP_WIDTH;
	int MAP_HEIGHT;

	/////particle generator type related////
	enum ParticleGeneratorType TYPE;

} simpleParticleSystem;

void init_particles_cuda(simpleParticleSystem &sps, int image_width, int image_height);
void init_vortex_field(simpleParticleSystem &sps);
void destroy_particles_cuda(simpleParticleSystem &sps);
void generate_particles(int generate_size, float2 point_pos);
void copy_to_device_sps(simpleParticleSystem &sps);
void render_particles(uchar4* devPtr, int img_width, int img_height);
void updata_particles(int generate_size, float passed_time);