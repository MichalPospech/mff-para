#include "kernels.h"

#define MAX(x,y) (x > y ? x : y)
#define EDGE_BLOCK 64
#define POINT_BLOCK 64


__global__ void simulate_compulsion(const Point<double>* points, const Edge<std::uint32_t>* edges, const std::uint32_t* lengths, Point<double>* velocities, double compulsion_constant, double vertex_mass, double time_quantum)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	Edge<std::uint32_t> edge = edges[idx];
	double x_diff =  points[edge.p2].x-points[edge.p1].x;
	double y_diff = points[edge.p2].y - points[edge.p1].y;
	double dist_squared = x_diff*x_diff + y_diff*y_diff;
	double dist = sqrt(dist_squared);
	double force_coeff = (dist*compulsion_constant)/(lengths[idx]);
	double x_force = x_diff * force_coeff;
	double y_force = y_diff * force_coeff;
	atomicAdd(&(velocities[edge.p1].x), x_force*time_quantum/vertex_mass);
	atomicAdd(&(velocities[edge.p1].y), y_force*time_quantum/vertex_mass);
	atomicAdd(&(velocities[edge.p2].x), -x_force*time_quantum/vertex_mass);
	atomicAdd(&(velocities[edge.p2].y), -y_force*time_quantum/vertex_mass);	
}


__global__ void simulate_repulsion(const Point<double>* points,  Point<double>* velocities, std::size_t point_count, double repulsion_constant, double vertex_mass, double time_quantum){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double f_x = 0;
	double f_y = 0;
	for (int idx2 = 0; idx2 < point_count; idx2++)
	{
		double x_diff = points[idx2].x - points[idx].x;
		double y_diff = points[idx2].y - points[idx].y;
		double dist_squared = x_diff * x_diff + y_diff * y_diff;
		dist_squared = MAX(dist_squared,(double)0.0001);
		double f = repulsion_constant / (dist_squared * sqrt(dist_squared));
		f_y += (f * -y_diff);
		f_x += (f * -x_diff);
	}
	velocities[idx].x+= f_x * time_quantum / vertex_mass;
	velocities[idx].y+= f_y * time_quantum / vertex_mass;
}

__global__ void simulate_movement(Point<double>* points,  Point<double>* velocities, double time_quantum, double slowdown){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	velocities[idx].x *= (slowdown);
	velocities[idx].y *= (slowdown);
	points[idx].x+=velocities[idx].x*time_quantum;
	points[idx].y+=velocities[idx].y*time_quantum;
}

/*
 * This is how a kernel call should be wrapped in a regular function call,
 * so it can be easilly used in cpp-only code.
 */
void step(Point<double>* points, const Edge<std::uint32_t>* edges, const std::uint32_t* lengths, Point<double>* velocities, const  ModelParameters<double>& parameters, std::size_t point_count, std::size_t edge_count)
{
	simulate_repulsion<<<point_count/POINT_BLOCK, POINT_BLOCK>>>(points, velocities, point_count, parameters.vertexRepulsion, parameters.vertexMass, parameters.timeQuantum);
	simulate_compulsion<<<edge_count/EDGE_BLOCK, EDGE_BLOCK>>>(points, edges, lengths, velocities, parameters.edgeCompulsion, parameters.vertexMass, parameters.timeQuantum);
	simulate_movement<<<point_count/POINT_BLOCK, POINT_BLOCK>>>(points, velocities, parameters.timeQuantum, parameters.slowdown);
}
