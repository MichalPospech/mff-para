#ifndef CUDA_POTENTIAL_IMPLEMENTATION_HPP
#define CUDA_POTENTIAL_IMPLEMENTATION_HPP

#include "kernels.h"

#include <interface.hpp>
#include <data.hpp>

#include <cuda_runtime.h>

/*
 * Final implementation of the tested program.
 */
template <typename F = float, typename IDX_T = std::uint32_t, typename LEN_T = std::uint32_t>
class ProgramPotential : public IProgramPotential<F, IDX_T, LEN_T>
{
public:
	typedef F coord_t;		// Type of point coordinates.
	typedef coord_t real_t; // Type of additional float parameters.
	typedef IDX_T index_t;
	typedef LEN_T length_t;
	typedef Point<coord_t> point_t;
	typedef Edge<index_t> edge_t;

private:
	std::size_t points_count_;
	std::size_t edge_count_;

	edge_t *cup_edges_;
	LEN_T *cup_lengths_;
	point_t *cup_velocities_;
	index_t iter_num;
	point_t *cup_points_;

public:
	virtual void initialize(index_t points, const std::vector<edge_t> &edges, const std::vector<length_t> &lengths, index_t iterations)
	{
		points_count_ = points;
		edge_count_ = edges.size();
		CUCH(cudaSetDevice(0));

		CUCH(cudaMalloc(&cup_edges_, edges.size() * sizeof(edge_t)));
		CUCH(cudaMemcpyAsync(cup_edges_, edges.data(), edges.size(), cudaMemcpyKind::cudaMemcpyHostToDevice));

		CUCH(cudaMalloc(&cup_lengths_, lengths.size() * sizeof(LEN_T)));
		CUCH(cudaMemcpyAsync(cup_lengths_, lengths.data(), lengths.size(), cudaMemcpyKind::cudaMemcpyHostToDevice));

		CUCH(cudaMalloc(&cup_velocities_, points * sizeof(point_t)));
		CUCH(cudaMemsetAsync(cup_velocities_, 0, points * sizeof(point_t)));

		CUCH(cudaMalloc(&cup_points_, points * sizeof(point_t)));
		iter_num = 0;
	}

	virtual void iteration(std::vector<point_t> &points)
	{
		if (!iter_num)
		{
			CUCH(cudaMemcpy(cup_points_, points.data(), points.size(), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}
		step(cup_points_, cup_edges_, cup_lengths_, cup_velocities_, this->mParams, points_count_, edge_count_);
		++iter_num;
		CUCH(cudaMemcpy(points.data(), cup_points_, points.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	}

	virtual void getVelocities(std::vector<point_t> &velocities)
	{
		velocities.reserve(points_count_);
		CUCH(cudaMemcpy(velocities.data(), cup_velocities_, points_count_, cudaMemcpyKind::cudaMemcpyDeviceToHost));
	}
};

#endif
