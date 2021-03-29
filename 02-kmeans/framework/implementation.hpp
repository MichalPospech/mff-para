#ifndef KMEANS_IMPLEMENTATION_HPP
#define KMEANS_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>
#include "tbb/tbb.h"
#include <math.h>
#include <iostream>

template <typename POINT = point_t, typename ASGN = std::uint8_t, bool DEBUG = false>
class KMeans : public IKMeans<POINT, ASGN, DEBUG>
{
	using coord_t = typename POINT::coord_t;
	std::vector<coord_t> distances;
	std::size_t k;
	std::size_t points;

	coord_t distance(const POINT &point, const POINT &centroid)
	{
		std::int64_t dx = (std::int64_t)point.x - (std::int64_t)centroid.x;
		std::int64_t dy = (std::int64_t)point.y - (std::int64_t)centroid.y;
		return (coord_t)(dx * dx + dy * dy);
	}

	void assign_cluster(const std::vector<POINT> &centroids, const std::vector<POINT> &points, std::size_t point_index, std::vector<ASGN> &assignments)
	{
		auto point = points[point_index];
		coord_t minDist = distance(point, centroids[0]);
		std::size_t nearest = 0;
		for (std::size_t i = 1; i < centroids.size(); ++i) {
			coord_t dist = distance(point, centroids[i]);
			if (dist < minDist) {
				minDist = dist;
				nearest = i;
			}
		}
		assignments[point_index] = nearest;
	}

	class cluster
	{

	public:
		POINT original;
		std::size_t count;
		POINT total;
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
		const std::vector<POINT> *points;
		const std::vector<ASGN> *assignments;

	public:
		std::vector<cluster> clusters;
		reduce_chunk(std::size_t centroid_num, const std::vector<POINT> *points, const std::vector<ASGN> *assignments) : points(points), assignments(assignments), clusters(centroid_num) {}

		reduce_chunk(reduce_chunk &chunk, tbb::split) : points(chunk.points), assignments(chunk.assignments), clusters(chunk.clusters.size())
		{
			for (std::size_t i = 0; i < clusters.size(); i++)
			{
				clusters[i].original = chunk.clusters[i].original;
			}
		}
		template <typename TRange>
		void operator()(const TRange &range)
		{
			for (auto i = range.begin(); i != range.end(); ++i)
			{
				/clusters[(*assignments)[i]].add((*points)[i]);
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
		distances = std::vector<coord_t>(points * k);
		this->k = k;
		this->points = points;
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
		centroids.resize(k);
		std::copy(points.begin(), points.begin() + k, centroids.begin());
		assignments.resize(points.size());
		for (std::size_t i = 0; i < iters; i++)
		{
			tbb::parallel_for(tbb::blocked_range<std::size_t>(0, points.size(), 1024), [&](auto &&range) {
				for (auto i = range.begin(); i != range.end(); ++i)
					assign_cluster(centroids, points, i, assignments);
			});
			auto reducer = reduce_chunk(k, &points, &assignments);
			tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, points.size(), 1024), reducer);
			for (size_t i = 0; i < k; i++)
			{
				centroids[i] = reducer.clusters[i].get_new_centroid();
			}
		}
	}
};

#endif
