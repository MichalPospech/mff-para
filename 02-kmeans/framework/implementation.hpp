#ifndef KMEANS_IMPLEMENTATION_HPP
#define KMEANS_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>
#include "tbb/tbb.h"
#include <math.h>

template <typename POINT = point_t, typename ASGN = std::uint8_t, bool DEBUG = false>
class KMeans : public IKMeans<POINT, ASGN, DEBUG>
{
	std::vector<double> distances;
	std::size_t k;
	std::size_t points;

	void assign_cluster(const std::vector<POINT> &centroids, const std::vector<POINT> &points, std::size_t point_index, std::vector<ASGN> &assignments)
	{
		auto offset = k * point_index;
		auto point = points[point_index];

		tbb::parallel_for(tbb::blocked_range<std::size_t>(0, centroids.size()), [&, offset, point](auto &&range) {
			for (auto i = range.begin(); i != range.end(); ++i)
				distances[offset + i] = std::sqrt(std::pow(centroids[i].x - point.x, 2.0) + std::pow(centroids[i].y - point.y, 2.0));
		});

		auto cluster_index = distances.begin() + offset - std::min_element(distances.begin() + offset, distances.begin() + offset + k);
		assignments[point_index] = cluster_index;
	}

	class cluster
	{
		POINT original;
		std::size_t count;
		POINT total;

	public:
		void add(const POINT &point)
		{
			total.x += point.x;
			total.y += point.y;
			++count;
		}

		POINT get_new_centroid()
		{
			POINT point = original;
			if (count)
			{
				point.x = total.x / count;
				point.y = total.y / count;
			}
			return point;
		}
	};

	class reduce_chunk
	{
		std::vector<cluster> clusters;

	public:
		reduce_chunk(reduce_chunk &chunk, tbb::split) : clusters(chunk.clusters.size())
		{
			for (std::size_t i = 0; i < clusters.size(); i++)
			{
				clusters[i].original += chunk.clusters[i].original;
			}
		}

		void operator()<TRange>(const TRange &range)
		{
			for (auto i = range.begin(); i != range.end(); ++i){
				
			}
				
		}

		void join(reduce_chunk &chunk)
		{
			for (std::size_t i = 0; i < clusters.size(); i++)
			{
				clusters[i].total.x += chunk.clusters[i].total.x;
				clusters[i].total.y += chunk.clusters[i].total.y;
				clusters[i].count += chunk.clusters[i].count;
			}
		}
	};

public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param points Number of points being clustered.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 */
	virtual void
	init(std::size_t points, std::size_t k, std::size_t iters)
	{
		distances = std::vector<double>(points * k);
	}

	/*
	 * \brief Perform the clustering and return the cluster centroids and point assignment
	 *		yielded by the last iteration.
	 * \note First k points are taken as initial centroids for first iteration.
	 * \param points Vector with input points.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 * \param centroids Vector where the final cluster centroids should be stored.
	 * \param assignments Vector where the final assignment of the points should be stored.
	 *		The indices should correspond to point indices in 'points' vector.
	 */
	virtual void compute(const std::vector<POINT> &points, std::size_t k, std::size_t iters,
						 std::vector<POINT> &centroids, std::vector<ASGN> &assignments)
	{
		for (std::size_t i = 0; i < iters; i++)
		{
			tbb::parallel_for(tbb::blocked_range<std::size_t>(0, points.size()), [&](auto &&range) {
				for (auto i = range.begin(); i != range.end(); ++i)
					assign_cluster(centroids, points, i, assignments);
			});
		}

		std::copy(points.begin(), points.begin() + k, centroids.begin());
		assign_cluster(centroids, points, 0, assignments);
		throw bpp::RuntimeError("Solution not implemented yet.");
	}
};

#endif
